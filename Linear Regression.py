#import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#import the dataset
data = pd.read_csv('Salary_Data.csv')
print(data.head())

print(data.describe()) #display basic statistics of the dataset
print(data.info()) #display information about the dataset
print(data.shape) #display the shape of the dataset

#plotting the data
plt.scatter(data['YearsExperience'], data['Salary'], color='blue')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary vs Years of Experience')
plt.show() #show the plot

#feature and target variable
X = data[['YearsExperience']].values  # Features (Years of Experience)
y = data['Salary'].values  # Target variable (Salary)
from sklearn.model_selection import train_test_split

#split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
from sklearn.linear_model import LinearRegression

#initialize the linear regression model
model = LinearRegression()

#fit the model on the training data
model.fit(X_train, y_train)

#make predictions on the test set
y_pred = model.predict(X_test)

#display the predictions
print("Predicted Salaries:", y_pred)

#display the actual salaries for comparison
print("Actual Salaries:", y_test)

#plotting the regression line
plt.scatter(X_train, y_train, color='blue', label='Training data')
plt.scatter(X_test, y_test, color='green', label='Test data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression line')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.title('Salary Prediction using Linear Regression')
plt.show()  # show the plot with regression line

from sklearn.metrics import mean_squared_error, r2_score

#calculate and display the performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R-squared:", r2)

print('Slope:', model.coef_)
print('Intercept:', model.intercept_)  # display the intercept of the regression line

#predicting a new value
new_experience = np.array([[5]])  # Example: Predicting salary for 5 years of experience
predicted_salary = model.predict(new_experience)
print(f"Predicted Salary for {new_experience[0][0]} years of experience: {predicted_salary[0]}")