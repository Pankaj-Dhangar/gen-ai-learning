import pandas as pd
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('data.csv')
print(data.info)

correlation = data['Speed'].corr(data['BrakingDistance'])
print(correlation)

covariance = data['Speed'].cov(data['BrakingDistance'])
print(covariance)

x = data.drop('BrakingDistance', axis=1)
y = data['BrakingDistance']

Model = LinearRegression()
Model.fit()
plot.scatter('Speed','BrakingDistance')
plot.xlabel('Speed')
plot.ylabel('BrakingDistance')
plot.title('Speed vs BrakingDistance')
plot.show()

