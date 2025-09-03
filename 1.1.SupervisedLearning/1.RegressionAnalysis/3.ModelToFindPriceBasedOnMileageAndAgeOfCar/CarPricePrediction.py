import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

df = pd.read_csv('car_data.csv', sep='\t')

model = LinearRegression()

x = df.drop('Price', axis=1)
y = df['Price']

model.fit(x, y)

input_data = pd.DataFrame([[5, 50000]], columns=['Age', 'Mileage'])

print(input_data)

predicted_price = model.predict(input_data)
print("Predicted Price for a car with 5 years of age and 50000 mileage :" + str(predicted_price[0]))