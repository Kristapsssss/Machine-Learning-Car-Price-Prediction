import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from scipy.stats import randint
import joblib
import os
import plotly.express as px
from keras.api.models import Sequential
from tensorflow.keras.layers import Dense
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.optimizers import Adam

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

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Neural Network': None,  # Placeholder for the neural network
    'Support Vector Regression': SVR(),
    'K-Nearest Neighbors': KNeighborsRegressor(),
    'Elastic Net': ElasticNet(),
    'XGBoost': XGBRegressor(),
    'LightGBM': LGBMRegressor()
}

# Hyperparameter grids
param_grids = {
    'Gradient Boosting': {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 15),
        'min_samples_split': randint(2, 15),
        'min_samples_leaf': randint(1, 10)
    },
    'Support Vector Regression': {
        'C': [0.1, 1, 10, 100],
        'epsilon': [0.01, 0.1, 0.5, 1]
    },
    'K-Nearest Neighbors': {
        'n_neighbors': randint(1, 20)
    },
    'Elastic Net': {
        'alpha': [0.1, 1, 10, 100],
        'l1_ratio': [0.1, 0.5, 0.7, 1.0]
    },
    'XGBoost': {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 15),
        'learning_rate': [0.01, 0.1, 0.3]
    },
    'LightGBM': {
        'n_estimators': randint(100, 300),
        'max_depth': randint(3, 15),
        'learning_rate': [0.01, 0.1, 0.3]
    }
}

# Function to create neural network model
def create_nn_model(optimizer='adam'):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))  # Output layer
    model.compile(loss='mean_absolute_error', optimizer=optimizer)
    return model

# Add the neural network to the models dictionary
models['Neural Network'] = KerasRegressor(build_fn=create_nn_model, epochs=50, batch_size=32, verbose=2)

# Train and evaluate models
model_stats = []

for model_name, model in models.items():
    if model_name in param_grids:
        random_search = RandomizedSearchCV(model, param_grids[model_name], n_iter=100, cv=5,
                                           scoring='neg_mean_absolute_error', n_jobs=-1, random_state=90, verbose=2)
        random_search.fit(X_train, y_train)
        best_model = random_search.best_estimator_
    elif model_name == 'Neural Network':
        model.fit(X_train, y_train)
        best_model = model
    else:
        model.fit(X_train, y_train)
        best_model = model

    # Save the model
    model_filename = f'{model_name.replace(" ", "_").lower()}_model.pkl'
    joblib.dump(best_model, model_filename)

    # Predictions
    y_pred = best_model.predict(X_test)

    # Evaluation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    model_stats.append({
        'Model Type': model_name,
        'Model Name': model_filename,
        'MAE': mae,
        'MSE': mse,
        'R²': r2
    })

    # Feature Importance for tree-based models
    if model_name in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
        importances = best_model.feature_importances_
        feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
        feature_importance_filename = f'{model_name.replace(" ", "_").lower()}_feature_importance.pkl'
        joblib.dump(feature_importance_df, feature_importance_filename)

        # Create feature importance bar chart
        fig_feature_importance = px.bar(feature_importance_df, x='Importance', y='Feature', orientation='h',
                                        title=f'{model_name} Feature Importance')
        fig_feature_importance.update_layout(yaxis={'categoryorder': 'total ascending'})
        fig_feature_importance.show()

# Save model stats
model_stats_df = pd.DataFrame(model_stats)
stats_file_path = 'model_stats.csv'

if os.path.exists(stats_file_path):
    existing_stats = pd.read_csv(stats_file_path)
    model_stats_df = pd.concat([existing_stats, model_stats_df], ignore_index=True)

model_stats_df.to_csv(stats_file_path, index=False)

# Compare models
print(model_stats_df)