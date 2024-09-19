import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from geopy.distance import great_circle
import folium
from folium.plugins import HeatMap
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the datasets
file_path_ds1 = 'data/nepal-earthquake-severity-index-latest.csv'
file_path_ds2 = 'data/earthquake_lat.csv'  # Update this to your actual file path

data_ds1 = pd.read_csv(file_path_ds1)
data_ds2 = pd.read_csv(file_path_ds2)

# Merge datasets based on a common key (assuming 'DISTRICT' is common in both datasets)
data = pd.merge(data_ds1, data_ds2, left_on='DISTRICT', right_on='Epicentre', how='left')

# Drop unnecessary columns for model training
data = data.drop(['P_CODE', 'VDC_NAME'], axis=1, errors='ignore')  # Ignore errors if columns do not exist

# Fill missing values with median (numerical) and mode (categorical)
for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])
    else:
        data[column] = data[column].fillna(data[column].median())

# Feature Engineering
data['Hazard_Level'] = pd.cut(data['Hazard (Intensity)'], bins=[0, 1.5, 2.5, 3], labels=['Low', 'Moderate', 'High'])
data['Housing_Vulnerability'] = np.where(data['Housing'] > data['Housing'].mean(), 1, 0)
data['Economic_Vulnerability'] = np.where(data['Poverty'] > data['Poverty'].mean(), 1, 0)
data['Composite_Vulnerability'] = data[['Vulnerability', 'Housing_Vulnerability', 'Economic_Vulnerability']].mean(axis=1)
data['Seismic_Zone'] = pd.cut(data['Hazard (Intensity)'], bins=[0, 1, 2, 3], labels=['Low Seismic Activity', 'Moderate Seismic Activity', 'High Seismic Activity'])
data['Resource_Need'] = (data['Poverty'] * data['Exposure'] * data['Hazard (Intensity)']).rank(method='max')
data['Damage_Prediction'] = (data['Hazard (Intensity)'] * data['Vulnerability'] * data['Housing']).rank(method='max')

# Assign Dispatch Priority based on the conditions
def assign_dispatch_priority(row):
    mean_vulnerability = data['Vulnerability'].mean()
    mean_severity = data['Severity'].mean()

    if row['Vulnerability'] > mean_vulnerability and row['Severity'] > mean_severity:
        return 'High'
    elif row['Vulnerability'] > mean_vulnerability or row['Severity'] > mean_severity:
        return 'Medium'
    else:
        return 'Low'

# Apply the function to create the Dispatch_Priority column
data['Dispatch_Priority'] = data.apply(assign_dispatch_priority, axis=1)

# Encoding dispatch priority using LabelEncoder
le = LabelEncoder()
data['Dispatch_Priority'] = le.fit_transform(data['Dispatch_Priority'])

# Select features and target variable
X = data[['Hazard (Intensity)', 'Vulnerability', 'Housing', 'Poverty', 'Exposure']]
y = data['Dispatch_Priority']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train a Random Forest Classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()

# Clustering using KMeans
X_clustering = data[['Composite_Vulnerability', 'Resource_Need', 'Damage_Prediction']]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Assign cluster labels to the data
data['Cluster'] = clusters

# Anomaly detection with Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)
data['Anomaly'] = anomalies

# Checking for anomalies
anomalies_detected = data[data['Anomaly'] == -1]
print("Anomalies detected in the following regions:\n", anomalies_detected[['DISTRICT', 'Anomaly']])

# Load Nepal's district shapefile for geospatial plotting
shapefile_path = 'nepal_map/gadm41_NPL_3.shp'
districts_gdf = gpd.read_file(shapefile_path)

# Merge earthquake data with shapefile
merged_data = districts_gdf.merge(data, left_on='NAME_3', right_on='DISTRICT', how='left')

# User input for latitude and longitude of the epicenter
latitude = float(input("Enter latitude of the epicenter: "))
longitude = float(input("Enter longitude of the epicenter: "))

# Function to calculate distance and adjust damage prediction
def adjust_damage_near_epicenter(df, epicenter_lat, epicenter_lon):
    df['Distance_to_Epicenter'] = df.apply(lambda row: great_circle((epicenter_lat, epicenter_lon), (row['Latitude'], row['Longitude'])).km, axis=1)
    max_distance = df['Distance_to_Epicenter'].max()
    df['Adjusted_Damage_Prediction'] = df['Damage_Prediction'] * (1 - (df['Distance_to_Epicenter'] / max_distance))
    return df

# Update the dataset with adjusted damage predictions
data = adjust_damage_near_epicenter(data, latitude, longitude)

# Re-merge updated data with shapefile
merged_data = districts_gdf.merge(data, left_on='NAME_3', right_on='DISTRICT', how='left')

# Plotting Adjusted Damage Prediction
plt.figure(figsize=(12, 10))
ax = merged_data.plot(column='Adjusted_Damage_Prediction', cmap='OrRd', legend=True,
                      legend_kwds={'label': "Adjusted Damage Prediction",
                                   'orientation': "horizontal"})

# Highlighting the epicenter
plt.scatter(longitude, latitude, color='blue', edgecolor='black', s=100, label='Epicenter')

plt.title(f'Heatmap of Adjusted Earthquake Damage Prediction by District (Epicenter at {latitude}, {longitude})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.show()

# Heatmap of earthquake intensities based on proximity to epicenter
def adjust_heatmap_weights(df, epicenter_lat, epicenter_lon):
    """
    Adjust heatmap intensity based on the distance from the epicenter.
    Points closer to the epicenter will have higher intensity (weights).
    """
    max_distance = df.apply(lambda row: great_circle((epicenter_lat, epicenter_lon), (row['Latitude'], row['Longitude'])).km, axis=1).max()
    
    # Calculate distance from the epicenter for each point
    df['Distance_to_Epicenter'] = df.apply(lambda row: great_circle((epicenter_lat, epicenter_lon), (row['Latitude'], row['Longitude'])).km, axis=1)
    
    # Calculate the weight (higher weight for points closer to the epicenter)
    df['Heatmap_Weight'] = 1 - (df['Distance_to_Epicenter'] / max_distance)
    
    # Normalize the weights to be between 0 and 1
    df['Heatmap_Weight'] = df['Heatmap_Weight'] / df['Heatmap_Weight'].max()
    
    return df

# Adjust the heatmap weights based on the epicenter
data = adjust_heatmap_weights(data, latitude, longitude)

# Create a Folium map centered on the epicenter
m = folium.Map(location=[latitude, longitude], zoom_start=7)

# Generate the heatmap data (latitude, longitude, weight)
heatmap_data = data[['Latitude', 'Longitude', 'Heatmap_Weight']].values.tolist()

# Add the heatmap layer to the folium map
HeatMap(heatmap_data, radius=15, max_zoom=13).add_to(m)

# Save the map to an HTML file and display it
m.save("earthquake_heatmap.html")
print("Heatmap saved as 'earthquake_heatmap.html'. You can open it in a browser to view the map.")

# Grid search for hyperparameter tuning
param_grid = {
    'n_estimators': [200],
    'max_depth': [30],
    'min_samples_split': [2]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Best hyperparameters: {grid_search.best_params_}")

