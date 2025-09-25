import pandas as pd
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures 

data = pd.read_csv('data.csv')
#print(data.info())

correlation = data['Speed'].corr(data['BrakingDistance'])
#print(correlation)

covariance = data['Speed'].cov(data['BrakingDistance'])
#print(covariance)

x = data.drop(columns='BrakingDistance', axis=1)
y = data['BrakingDistance']

ploy = PolynomialFeatures(degree=6)
x_Square = ploy.fit_transform(x)

#print (x_Square)
# y = mx
# y = c+mx*2

model = LinearRegression()
model.fit(x_Square,y)

output = model.predict(ploy.fit_transform([[120]]))
print(output)

# plot.scatter(data['Speed'], data['BrakingDistance'])
# plot.xlabel('Speed')
# plot.ylabel('BrakingDistance')
# plot.title('Speed vs Braking Distance')
# plot.show()