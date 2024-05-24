from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from scipy.stats import randint
import joblib

# Load and preprocess the data
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

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=90)

# Define the model
gb = GradientBoostingRegressor()

param_dist = {
    'n_estimators': randint(100, 300),
    'max_depth': randint(3, 15),
    'min_samples_split': randint(2, 15),
    'min_samples_leaf': randint(1, 10)
}

# Use RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(estimator=gb, param_distributions=param_dist, n_iter=100, cv=5,
                                   scoring='neg_mean_absolute_error', n_jobs=-1, random_state=90, verbose=2)

# Fit the model
random_search.fit(X_train, y_train)

# Best model
best_gb = random_search.best_estimator_

# Save the model
joblib.dump(best_gb, '../WEB-Portfolio/static/models/gradient_boosting_model.pkl')

# Save feature importance
importances = best_gb.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values(by='Importance', ascending=False)
joblib.dump(feature_importance_df, '../WEB-Portfolio/static/data/gradient_boosting_model_feature_importance.pkl')

# Evaluate the model
y_pred = best_gb.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MAE: {mae}, MSE: {mse}, R²: {r2}')