# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('nepal-earthquake-severity-index-latest.csv')

# 1. Handle Missing Values
# Numerical columns: Use mean imputation
numerical_columns = ['Hazard (Intensity)', 'Exposure', 'Housing', 'Poverty', 'Vulnerability', 'Severity', 'Severity Normalized']
imputer_num = SimpleImputer(strategy='mean')
df[numerical_columns] = imputer_num.fit_transform(df[numerical_columns])

# Categorical columns: Use mode imputation
categorical_columns = ['DISTRICT', 'REGION', 'Severity category']
imputer_cat = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = imputer_cat.fit_transform(df[categorical_columns])

# 2. Feature Engineering (Add new features based on existing ones)
df['Risk_Index'] = df['Exposure'] * df['Vulnerability']

# 3. Outlier Detection and Handling
# Detecting outliers using z-scores and capping them to a threshold
z_threshold = 3
for col in numerical_columns:
    col_zscore = (df[col] - df[col].mean()) / df[col].std()
    df[col] = np.where(col_zscore > z_threshold, df[col].mean() + z_threshold * df[col].std(), df[col])
    df[col] = np.where(col_zscore < -z_threshold, df[col].mean() - z_threshold * df[col].std(), df[col])

# 4. Feature Scaling
# Using RobustScaler for scaling to handle outliers better
scaler = RobustScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

# 5. Encoding Categorical Variables
label_encoder = LabelEncoder()
df['DISTRICT'] = label_encoder.fit_transform(df['DISTRICT'])
df['REGION'] = label_encoder.fit_transform(df['REGION'])
df['Severity category'] = label_encoder.fit_transform(df['Severity category'])

# 6. Train-Test Split (Stratified)
X = df.drop(columns=['P_CODE', 'VDC_NAME', 'Severity', 'Severity category'])
y = df['Severity']

# Count the occurrences of each class in y
class_counts = y.value_counts()

# Filter classes with less than 2 samples
filtered_classes = class_counts[class_counts > 1].index

# Keep only rows where y belongs to classes with more than 1 sample
X_filtered = X[y.isin(filtered_classes)]
y_filtered = y[y.isin(filtered_classes)]

# Now you can do stratified split
X_train, X_test, y_train, y_test = train_test_split(X_filtered, y_filtered, test_size=0.2, random_state=42, stratify=y_filtered)

# 7. Seismic Pattern Recognition (K-Means Clustering)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train)

# Assign clusters to the training data only
X_train['Cluster'] = kmeans.labels_

# Predict clusters for the entire dataset for visualization purposes
df['Cluster'] = kmeans.predict(X)

# Visualization of clusters (using the whole dataset)
sns.scatterplot(x=df['Hazard (Intensity)'], y=df['Severity'], hue=df['Cluster'])
plt.title('Seismic Pattern Clustering')
plt.show()

# 8. Vulnerability Dispatch (RandomForest Regressor for Severity)
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train.drop(columns=['Cluster']), y_train)

X_test['Cluster'] = kmeans.predict(X_test)

# Now you can use the regressor and drop the 'Cluster' column if needed
y_pred_reg = regressor.predict(X_test.drop(columns=['Cluster']))

# MSE for Vulnerability Dispatch (severity prediction)
mse = mean_squared_error(y_test, y_pred_reg)
print(f'Vulnerability Dispatch MSE: {mse}')


# 9. Resource Allocation (RandomForest Regressor)
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train.drop(columns=['Cluster']), y_train)

# MSE for Resource Allocation
y_pred_reg = regressor.predict(X_test.drop(columns=['Cluster']))
mse = mean_squared_error(y_test, y_pred_reg)
print(f'Resource Allocation MSE: {mse}')

# 10. Urban vs. Rural Comparison (Analysis)
df['Urban_Rural'] = df['DISTRICT'].apply(lambda x: 'Urban' if x < 10 else 'Rural')
urban_data = df[df['Urban_Rural'] == 'Urban']
rural_data = df[df['Urban_Rural'] == 'Rural']

# Severity comparison
plt.figure(figsize=(10, 6))
sns.boxplot(x='Urban_Rural', y='Severity', data=df)
plt.title('Severity Comparison: Urban vs Rural')
plt.show()
