import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
import geopandas as gpd

file_path = 'data/nepal-earthquake-severity-index-latest.csv'
data = pd.read_csv(file_path)

data = data.drop(['P_CODE', 'VDC_NAME'], axis=1)

for column in data.columns:
    if data[column].dtype == 'object':
        data[column] = data[column].fillna(data[column].mode()[0])
    else:
        data[column] = data[column].fillna(data[column].median())

data['Hazard_Level'] = pd.cut(data['Hazard (Intensity)'], bins=[0, 1.5, 2.5, 3], labels=['Low', 'Moderate', 'High'])
data['Housing_Vulnerability'] = np.where(data['Housing'] > data['Housing'].mean(), 1, 0)
data['Economic_Vulnerability'] = np.where(data['Poverty'] > data['Poverty'].mean(), 1, 0)
data['Composite_Vulnerability'] = data[['Vulnerability', 'Housing_Vulnerability', 'Economic_Vulnerability']].mean(axis=1)
data['Seismic_Zone'] = pd.cut(data['Hazard (Intensity)'], bins=[0, 1, 2, 3], labels=['Low Seismic Activity', 'Moderate Seismic Activity', 'High Seismic Activity'])
data['Resource_Need'] = (data['Poverty'] * data['Exposure'] * data['Hazard (Intensity)']).rank(method='max')
data['Damage_Prediction'] = (data['Hazard (Intensity)'] * data['Vulnerability'] * data['Housing']).rank(method='max')


def assign_dispatch_priority(row):
    mean_vulnerability = data['Vulnerability'].mean()
    mean_severity = data['Severity'].mean()
    
    if row['Vulnerability'] > mean_vulnerability and row['Severity'] > mean_severity:
        return 'High'
    elif row['Vulnerability'] > mean_vulnerability or row['Severity'] > mean_severity:
        return 'Medium'
    else:
        return 'Low'

data['Dispatch_Priority'] = data.apply(assign_dispatch_priority, axis=1)

le = LabelEncoder()
data['Dispatch_Priority'] = le.fit_transform(data['Dispatch_Priority'])

X = data[['Hazard (Intensity)', 'Vulnerability', 'Housing', 'Poverty', 'Exposure']]
y = data['Dispatch_Priority']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d")
plt.show()

X_clustering = data[['Composite_Vulnerability', 'Resource_Need', 'Damage_Prediction']]
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_clustering)

kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_pca)

data['Cluster'] = clusters

iso_forest = IsolationForest(contamination=0.1, random_state=42)
anomalies = iso_forest.fit_predict(X_scaled)
data['Anomaly'] = anomalies

anomalies_detected = data[data['Anomaly'] == -1]
print("Anomalies detected in the following regions:\n", anomalies_detected[['DISTRICT', 'Anomaly']])

shapefile_path = 'nepal_map/gadm41_NPL_3.shp'
districts_gdf = gpd.read_file(shapefile_path)

merged_data = districts_gdf.merge(data, left_on='NAME_3', right_on='DISTRICT', how='left')

plt.figure(figsize=(12, 10))
ax = merged_data.plot(column='Damage_Prediction', cmap='OrRd', legend=True,
                      legend_kwds={'label': "Damage Prediction",
                                   'orientation': "horizontal"})

plt.title('Heatmap of Earthquake Damage Prediction by District')
plt.axis('off')
plt.show()

param_grid = {
    'n_estimators': [200],
    'max_depth': [30],
    'min_samples_split': [2]
}

grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")

grouped_data = data.groupby('REGION').agg({
    'Composite_Vulnerability': 'mean',
    'Resource_Need': 'mean',
    'Damage_Prediction': 'mean',
    'Dispatch_Priority': lambda x: x.mode()[0] 
}).reset_index()

def categorize_vulnerability(vuln):
    if vuln < 1.0:
        return "Low"
    elif vuln < 2.0:
        return "Moderate"
    else:
        return "High"

def categorize_resource_need(resource):
    if resource < data['Resource_Need'].quantile(0.33):
        return "Low"
    elif resource < data['Resource_Need'].quantile(0.66):
        return "Moderate"
    else:
        return "High"

def categorize_damage_prediction(damage):
    if damage < data['Damage_Prediction'].quantile(0.33):
        return "Low"
    elif damage < data['Damage_Prediction'].quantile(0.66):
        return "Moderate"
    else:
        return "High"

priority_mapping = {'High': 0, 'Medium': 1, 'Low': 2}
data['Dispatch_Priority'] = data['Dispatch_Priority'].map(priority_mapping)

for index, row in grouped_data.iterrows():
    region = row['REGION']
    vulnerability = row['Composite_Vulnerability']
    resource_need = row['Resource_Need']
    damage_prediction = row['Damage_Prediction']
    dispatch_priority = row['Dispatch_Priority']
    
    vuln_label = categorize_vulnerability(vulnerability)
    resource_label = categorize_resource_need(resource_need)
    damage_label = categorize_damage_prediction(damage_prediction)
    if dispatch_priority == 0:
        priority_level = "High"
    elif dispatch_priority == 1:
        priority_level = "Moderate"
    else:
        priority_level = "Low"
    
    print(f"\nRegion: {region}")
    print(f" - Composite Vulnerability: {vulnerability:.2f} ({vuln_label})")
    print(f" - Resource Need: {resource_need:.2f} ({resource_label})")
    print(f" - Damage Prediction: {damage_prediction:.2f} ({damage_label})")
    print(f" - Dispatch Priority: {dispatch_priority} ({priority_level})")
    
    if dispatch_priority == 0: 
        print(f"   -> This region needs urgent help due to high vulnerability and priority.")
    elif dispatch_priority == 1: 
        print(f"   -> This region has moderate vulnerability and may require additional support.")
    else: 
        print(f"   -> This region has low priority and vulnerability for now.")
    
    if resource_label == "High":
        print(f"   -> More resources should be allocated to this region as it has high resource need.")
