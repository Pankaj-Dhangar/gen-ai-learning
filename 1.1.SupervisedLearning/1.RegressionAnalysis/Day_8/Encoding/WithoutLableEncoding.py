import pandas as pd 
from sklearn.linear_model import LinearRegression

df = pd.read_csv('Salary_Data_transformed.csv')

x = df.drop('Salary', axis=1)
y = df['Salary']

model = LinearRegression()
model.fit(x,y)

output = model.predict(pd.DataFrame([[1,8]],columns=['Title', 'Experience']))
print(output)