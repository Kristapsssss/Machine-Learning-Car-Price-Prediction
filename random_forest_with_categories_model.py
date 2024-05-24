from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
from scipy.stats import randint
import plotly.express as px
import joblib
import os

# Load your data
data = pd.read_pickle('latvia_car_data.pkl')

# Filter models with more than 50 data points
model_counts = data['Model'].value_counts()
sufficient_data_models = model_counts[model_counts > 50].index
filtered_data = data[data['Model'].isin(sufficient_data_models)]

# Create new feature 'Car Age'
filtered_data['Car Age'] = 2024 - filtered_data['Year']

# Preprocess the data
filtered_data = filtered_data.dropna()
filtered_data = pd.get_dummies(filtered_data, columns=['Brand', 'Model', 'Engine Type'])  # Encoding categorical variables

# Split the data
X = filtered_data.drop('Price', axis=1)  # Features
y = filtered_data['Price']  # Target variable

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=90)

# Define the model
rf = RandomForestRegressor()

param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(10, 30),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10),
    'bootstrap': [True, False]
}

random_search = RandomizedSearchCV(estimator=rf, param_distributions=param_dist, n_iter=100, cv=5,
                                   scoring='neg_mean_absolute_error', n_jobs=-1, random_state=90, verbose=2)

random_search.fit(X_train, y_train)

# Best model
best_rf = random_search.best_estimator_

# Cross-validation score
cv_score = cross_val_score(best_rf, X_train, y_train, cv=5, scoring='neg_mean_absolute_error')
print(f'Cross-validation MAE: {-cv_score.mean()}')

# Predictions
y_pred = best_rf.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}, MSE: {mse}, R²: {r2}')

# Feature Importance
importances = best_rf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
print(feature_importance_df)

# # Save the model
# joblib.dump(best_rf, 'random_forest_model.pkl')
loaded_rf = joblib.load('Models/random_forest_model.pkl')

# # Save Feature Importance
# joblib.dump(feature_importance_df, 'feature_importance_df.pkl')
loaded_feature_importance_df = joblib.load('Feature Importance/feature_importance_df.pkl')


# Predict using the loaded model
y_pred_loaded = loaded_rf.predict(X_test)

# Calculate Model Stats
mae = mean_absolute_error(y_test, y_pred_loaded)
mse = mean_squared_error(y_test, y_pred_loaded)
r2 = r2_score(y_test, y_pred_loaded)

# Save Model Stats
model_stats = pd.DataFrame({
    'Model Type': ['Random Forest With Categories (Brand, Model, Engine Type)'],
    'Model Name': ['random_forest_model.pkl'],
    'MAE': [mae],
    'MSE': [mse],
    'R²': [r2]
})

# Define the stats file path
stats_file_path = 'model_stats.csv'

# Check if the file exists, and append if it does
if os.path.exists(stats_file_path):
    existing_stats = pd.read_csv(stats_file_path)
    model_stats = pd.concat([existing_stats, model_stats], ignore_index=True)

# Save the updated stats
model_stats.to_csv(stats_file_path, index=False)



