import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet, RANSACRegressor
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from joblib import dump, load

# Load the dataset
df = pd.read_csv('Car_Purchasing_Data.csv')

# Display basic information about the dataset
print(df.head())
print(df.tail())
print(df.shape)
print(df.info())

# Check for null values in the DataFrame
print(df.isnull().sum())

# Create input dataset from original dataset by dropping irrelevant features
X = df.drop(['Customer Name', 'Customer e-mail', 'Country', 'Car Purchase Amount'], axis=1)
Y = df['Car Purchase Amount']

# Transform input dataset into percentage-based weighted between 0 and 1
sc = MinMaxScaler()
x_scaled = sc.fit_transform(X)

# Transform output dataset into percentage-based weighted between 0 and 1
sc1 = MinMaxScaler()
y_reshape = Y.values.reshape(-1, 1)
y_scaled = sc1.fit_transform(y_reshape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.2, train_size=0.8, random_state=42, shuffle=False)

# Initialize the models
linear_model = LinearRegression()
svm = SVR()
rf = RandomForestRegressor()
gbr = GradientBoostingRegressor()
ridge = Ridge()
en = ElasticNet()
rr = RANSACRegressor()
dtr = DecisionTreeRegressor()
ann = MLPRegressor(max_iter=1000)
etr = ExtraTreesRegressor()

# Train the models using the training set
linear_model.fit(X_train, y_train)
svm.fit(X_train, y_train)
rf.fit(X_train, y_train)
gbr.fit(X_train, y_train)
ridge.fit(X_train, y_train)
en.fit(X_train, y_train)
rr.fit(X_train, y_train)
dtr.fit(X_train, y_train)
ann.fit(X_train, y_train)
etr.fit(X_train, y_train)

# Prediction on the validation/test data
linear_model_preds = linear_model.predict(X_test)
svm_preds = svm.predict(X_test)
rf_preds = rf.predict(X_test)
gbr_preds = gbr.predict(X_test)
ridge_preds = ridge.predict(X_test)
en_preds = en.predict(X_test)
rr_preds = rr.predict(X_test)
dtr_preds = dtr.predict(X_test)
ann_preds = ann.predict(X_test)
etr_preds = etr.predict(X_test)

# Evaluate model performance
linear_model_rmse = mean_squared_error(y_test, linear_model_preds, squared=False)
svm_rmse = mean_squared_error(y_test, svm_preds, squared=False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared=False)
gbr_rmse = mean_squared_error(y_test, gbr_preds, squared=False)
ridge_rmse = mean_squared_error(y_test, ridge_preds, squared=False)
en_rmse = mean_squared_error(y_test, en_preds, squared=False)
rr_rmse = mean_squared_error(y_test, rr_preds, squared=False)
dtr_rmse = mean_squared_error(y_test, dtr_preds, squared=False)
ann_rmse = mean_squared_error(y_test, ann_preds, squared=False)
etr_rmse = mean_squared_error(y_test, etr_preds, squared=False)

# Display the evaluation results
print(f"Linear Regression RMSE: {linear_model_rmse}")
print(f"Support Vector Machine RMSE: {svm_rmse}")
print(f"Random Forest Regressor RMSE: {rf_rmse}")
print(f"Gradient Boosting Regressor RMSE: {gbr_rmse}")
print(f"Ridge Regression RMSE: {ridge_rmse}")
print(f"Elastic Net RMSE: {en_rmse}")
print(f"Robust Regression RMSE: {rr_rmse}")
print(f"Decision Tree Regressor RMSE: {dtr_rmse}")
print(f"Artificial Neural Network RMSE: {ann_rmse}")
print(f"Extra Trees Regressor RMSE: {etr_rmse}")

# Choose the best model
model_objects = [linear_model, svm, rf, gbr, ridge, en, rr, dtr, ann, etr]
rmse_value = [linear_model_rmse, svm_rmse, rf_rmse, gbr_rmse, ridge_rmse, en_rmse, rr_rmse, dtr_rmse, ann_rmse, etr_rmse]

best_model_index = rmse_value.index(min(rmse_value))
best_model_object = model_objects[best_model_index]

# Visualize the model results
models = ['Linear Regression', 'Support Vector Machine', 'Random Forest Regressor', 'Gradient Boosting Regressor', 'Ridge Regression', 'Elastic Net', 'Robust Regression', 'Decision Tree Regressor', 'Artificial Neural Network', 'Extra Trees Regressor']

plt.figure(figsize=(10, 7))
bars = plt.bar(models, rmse_value, color=['blue', 'green', 'orange', 'red', 'yellow', 'grey', 'purple', 'pink', 'black', 'brown'])

# Add RMSE values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.00001, round(yval, 10), ha='center', va='bottom', fontsize=10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparison')
plt.xticks(rotation=45)  # Rotate model names for better visibility
plt.tight_layout()
plt.show()

# Retrain the best model on the entire dataset
best_model_object.fit(x_scaled, y_scaled)

# Save the best model
dump(best_model_object, "car_model.joblib")

# Load the model
loaded_model = load('car_model.joblib')

# Gathering user inputs
try:
    gender = int(input("Enter gender (0 for female, 1 for male): "))
    age = float(input("Enter age: "))
    annual_salary = float(input("Enter annual salary: "))
    credit_card_debt = float(input("Enter credit card debt: "))
    net_worth = float(input("Enter net worth: "))
except ValueError as e:
    print(f"Invalid input: {e}")
    exit()

# Ensure the new input data has the same feature names as the original data
feature_names = ['Gender', 'Age', 'Annual Salary', 'Credit Card Debt', 'Net Worth']
input_data = pd.DataFrame([[gender, age, annual_salary, credit_card_debt, net_worth]], columns=feature_names)

# Transform the new input data
X_test1 = sc.transform(input_data)

# Predict on new test data
preds_value = loaded_model.predict(X_test1)
predicted_amount = sc1.inverse_transform(preds_value)
print("Predicted Car Purchase Amount based on input: ", predicted_amount[0][0])
