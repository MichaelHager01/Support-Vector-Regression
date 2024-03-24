import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.svm import SVR



# Loads data from file steel_strength.csv
raw_data = pd.read_csv("steel_strength.csv")
features = raw_data[['c', 'mn', 'si', 'cr', 'ni', 'mo', 'v', 'n', 'nb', 'co', 'w', 'al', 'ti']].values
response = raw_data[['yield strength', 'tensile strength', 'elongation']].values
yield_strength = raw_data['yield strength'].values
tensile_strength = raw_data['tensile strength'].values
elongation = raw_data['elongation'].values


# Randomly splits the data into two subsets: 75% for training and 25% for test
X_train, X_test, Y_train, Y_test = train_test_split(features, tensile_strength, test_size=0.25, random_state=42)


# Defines a search grid for hyperparameter tuning via GridSearchCV
params = {
    'C': [0.1, 1, 10, 100, 250, 500, 1000],
    'gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
}


# Conducts hyperparameter using GridSearchCV for an SVR model with an RBF kernel and epsilon of 50
svr = SVR(kernel='rbf', epsilon=50)
grid_search = GridSearchCV(svr, params, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, Y_train)


# Best hyperparameters
best_params = grid_search.best_params_


# Predictions on test set
Y_pred = grid_search.predict(X_test)


# Compute RMSE and r2
rmse_train = mean_squared_error(Y_train, grid_search.predict(X_train), squared=False)
rmse_test = mean_squared_error(Y_test, Y_pred, squared=False)
r2 = r2_score(Y_test, Y_pred)



# Plot true vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(Y_test, Y_pred)
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], 'k--')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('True vs Predicted Values')
plt.grid(True)
plt.show()


# Print RMSE and r2
print()
print(f"RMSE on training set:   {rmse_train}")
print(f"RMSE on test set:       {rmse_test}")
print(f"r2 on test set:         {r2}")


# Define best hyperparameters obtained previously
best_params = {'C': 1000, 'gamma': 0.05}


# Create SVR model with best hyperparameters
svr_best_50 = SVR(kernel='rbf', epsilon=50, **best_params)
svr_best_100 = SVR(kernel='rbf', epsilon=100, **best_params)


# Fit the model
svr_best_50.fit(X_train, Y_train)
svr_best_100.fit(X_train, Y_train)


# Get the number of support vectors
n_support_vectors_best_50 = svr_best_50.support_vectors_.shape[0]
n_support_vectors_best_100 = svr_best_100.support_vectors_.shape[0]
print("Number of support vectors with best hyperparameter values with epsilon of 50:", n_support_vectors_best_50)
print("Number of support vectors with best hyperparameter values with epsilon of 100:", n_support_vectors_best_100)

