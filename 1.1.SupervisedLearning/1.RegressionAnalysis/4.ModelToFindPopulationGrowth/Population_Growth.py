import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = pd.read_csv('population_growth_data_1990_2025.csv')
#print(data.info())

correlation = data['Year'].corr(data['Population_Millions'])
#print(correlation)

covariance = data['Year'].cov(data['Population_Millions'])
#print(covariance)

x = data.drop(columns='Population_Millions', axis=1)
y= data['Population_Millions']

poly = PolynomialFeatures(degree=80)
x_poly = poly.fit_transform(x)

model = LinearRegression()
model.fit(x_poly, y)

output = model.predict(poly.fit_transform([[2025]]))
print(output)

#plt.scatter(data['Year'], data['Population_Millions'])
#plt.xlabel('Year')
#plt.ylabel('Population Millions')
#plt.title('Year vs Population Millions')
#plt.show()



