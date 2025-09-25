import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

df = pd.read_csv('salary_data.csv', sep="\t")

plt.scatter(df['Experience'], df['Salary'])
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Experience vs Salary")
#plt.show()

x = df.drop('Salary',axis=1)
y= df['Salary']
model = LinearRegression()
model.fit(x,y)

re_predict_all_x = model.predict(x)

output = model.predict([[11]]) # predict salary for 5 year of exp
print(output)

# since we have model which is trained, we can plot Best fit regression line

plt.scatter(df['Experience'], df['Salary'], label ='Observed data')
plt.scatter(df['Experience'], re_predict_all_x,color = 'blue', label='Predicted data')
plt.plot(df['Experience'], re_predict_all_x,color = 'red',label ='BEST FIT LINE')
plt.legend()

score = model.score(x,y) # show the score of model
print("score :{score}")

plt.show()