import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Ridge, ElasticNet,RANSACRegressor, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor,ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
#from xgboost import XGBRegressor
from joblib import dump
from joblib import load

#1. Linear Regression
#2. Robust Regression
#3. Ridge Regression
#4. Binary Logistic Regression
#5. Elastic Net
#6. Polynomial Regression
#7. Ordinal Logistic Regression
#8. Artificial Neural Networks (ANNs)
#9. Random Forest Regressor
#10. Support Vector Machine
df = pd.read_csv('Car_Purchasing_Data.csv')

# # Display the first 5 rows of the DataFrame
# print(df.head())

# # Display the last 5 rows of the DataFrame
# print(df.tail())

# # Get the shape of the DataFrame
# shape = df.shape
# print(f'The dataset has {shape[0]} rows and {shape[1]} columns.')

# df.info()


# # Check for null values in the DataFrame
# null_values = df.isnull().sum()
# print(null_values)

# #Identify library to plot graph to understand relations among various column

# sns.scatterplot(x='Customer Name', y='Country', data=df)
# plt.title('Relationship between Column 1 and Column 2')
# plt.show()

#Create input dataset from original dataset by dropping irrelevant features
#Dropped colomn will be stored in variable X
#axis=1: this is parameter specifies the axis along which to drop the columns
#in this case, axis=1 means that you want to drop colomns (as opposed to row, which would be axis=0)
X = df.drop (['Customer Name','Customer e-mail', 'Country','Car Purchase Amount'], axis=1)
# print (f"Dropped columns from dataset is {X}")

#create output dataset from original dataset
#output colomn will be store in variable Y
Y= df['Car Purchase Amount']
# print (f"output column from dataset is {Y}")

#Transform input dataset into percentage based weighted between 0 and 1
sc= MinMaxScaler()
x_scaled = sc.fit_transform(X)
print(x_scaled)

#Transform output dataset into percentage based weighted between 0 and 1
sc1 = MinMaxScaler() 
y_reshape = Y.values.reshape(-1,1)
y_scaled = sc1.fit_transform(y_reshape)
print(y_scaled)

#Print first few rows of scaled input dataset
print(x_scaled[:5])

#Print first few rows of scaled output dataset
print(y_scaled[:5])

#split data into training and testing
X_train, X_test, y_train, y_test = train_test_split (x_scaled,y_scaled, test_size=0.2, train_size=0.8, random_state=42, shuffle=False)

print("X_train:\n", X_train)
print("X_test:\n", X_test)
print("y_train:\n", y_train)
print("y_test:\n", y_test)

# Print the shapes of the resulting splits
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Print the first few rows of each
print("\nFirst few rows of X_train:\n", X_train[:5])
print("First few rows of X_test:\n", X_test[:5])
print("First few rows of y_train:\n", y_train[:5])
print("First few rows of y_test:\n", y_test[:5])

# Initialize the model
#model used: Linear Regress, Support Vector Machine, Random Forest Regressor,Gradient Boosting Regressor,Ridge Regression, Elastic Net, Robust Regression, Decision Tree Regressor, Artificial Neural Network,Extra Trees Regressor
linear_model = LinearRegression()
svm= SVR()
rf= RandomForestRegressor()
gbr = GradientBoostingRegressor()
ridge= Ridge()
en= ElasticNet()
rr= RANSACRegressor()
dtr= DecisionTreeRegressor()
ann=  MLPRegressor(max_iter=1000)
etr= ExtraTreesRegressor()


#train the models using training set
linear_model.fit (X_train, y_train)
svm.fit(X_train,y_train)
rf.fit (X_train,y_train)
gbr.fit (X_train,y_train)
ridge.fit (X_train,y_train)
en.fit (X_train,y_train)
rr.fit (X_train,y_train)
dtr.fit (X_train,y_train)
ann.fit (X_train,y_train)
etr.fit (X_train,y_train)


#prediction on the validation / test data
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

#evaluate model performance
#RMSE is a measure of the difference between the prdicted values by the model and the actual values
linear_model_rmse = mean_squared_error(y_test, linear_model_preds, squared= False)
svm_rmse = mean_squared_error(y_test, svm_preds, squared= False)
rf_rmse = mean_squared_error(y_test, rf_preds, squared= False)
gbr_rmse = mean_squared_error(y_test, gbr_preds, squared= False)
ridge_rmse = mean_squared_error(y_test, ridge_preds, squared= False)
en_rmse = mean_squared_error(y_test, en_preds, squared= False)
rr_rmse = mean_squared_error(y_test, rr_preds, squared= False)
dtr_rmse = mean_squared_error(y_test, dtr_preds, squared= False)
ann_rmse = mean_squared_error(y_test, ann_preds, squared= False)
etr_rmse = mean_squared_error(y_test, etr_preds, squared= False)

#display the evaluation results
#model used: Linear Regress, Support Vector Machine, Random Forest Regressor,Gradient Boosting Regressor,Ridge Regression, Elastic Net, Robust Regression, Decision Tree Regressor, Artificial Neural Network,Extra Trees Regressor
print(f"Linear Regression RMSE : {linear_model_rmse}")
print(f"Support Vector Machine RMSE : {svm_rmse}")
print(f"Randome Forest Regressor : {rf_rmse}")
print(f"Gradient Boosting Regressor : {gbr_rmse}")
print(f"Ridge Regression : {ridge_rmse}")
print (f"Elastic Net : {en_rmse}")
print(f"Robust Regression :{rr_rmse}")
print(f"Decision Tree Regressor : {dtr_rmse}")
print(f"Artificial Neural Network : {ann_rmse}")
print(f"Extra Trees Regressor : {etr_rmse}")





#CHOOSE the best model

model_objects = [linear_model,svm, rf, gbr,ridge,en,rr,dtr,ann,etr]
rmse_value = [linear_model_rmse,svm_rmse,rf_rmse,gbr_rmse,ridge_rmse,en_rmse,rr_rmse,dtr_rmse,ann_rmse,etr_rmse]

best_model_index = rmse_value.index(min(rmse_value))
best_model_object = model_objects[best_model_index]

#visualize the model results
#create a bar chart
models = ['Linear Regress', 'Support Vector Machine', 'Random Forest Regressor','Gradient Boosting Regressor','Ridge Regression', 'Elastic Net', 'Robust Regression', 'Decision Tree Regressor', 'Artificial Neural Network','Extra Trees Regressor']

plt.figure(figsize=(10,7))
bars = plt.bar (models, rmse_value, color = ['blue', 'green', 'orange', 'red', 'yellow', 'grey','purple', 'pink','black', 'brown'])

#add rmse values on top of each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x()+ bar.get_width()/2, yval + 0.00001, round(yval, 10), ha= 'center', va='bottom', fontsize = 10)

plt.xlabel('Models')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.title('Model RMSE Comparision')
plt.xticks(rotation = 45) #rotate model names for better visibility
plt.tight_layout()
#display the chart
plt.show()
#linear regression is the best model



#retrain the model on entire dataset
linear_model_final = LinearRegression()
linear_model_final.fit(x_scaled, y_scaled)


# #save the model
# Define the filename for the saved model
# model_filename = 'linear_regression_model.sav'

# Save the model to the file
dump(best_model_object, "car_model.joblib")

# print(f"Model saved as '{model_filename}'")

#load the model
# Define the filename from which to load the model
# model_filename = 'linear_regression_model.sav'

# Load the model from the file
loaded_model = load('car_model.joblib')

# print(f"Model loaded from '{model_filename}'")

#gathering user inputs


# Gather user inputs
gender = int(input("Enter gender( 0 for female, 1 for male) :"))
age = float(input("Enter age: "))
annual_salary = float(input("Enter annual salary: "))
credit_card_debt = float(input("Enter credit card debt: "))
net_worth = float(input("Enter net worth: "))

# Make predictions using the loaded model
X_test1 = sc.transform ([[gender,age, annual_salary, credit_card_debt, net_worth]])

#prediect on new test data
preds_value = loaded_model.predict (X_test1)
print(preds_value)
print("Predicted Car_Purchase_Amount based on input: ", sc1.inverse_transform(preds_value))
