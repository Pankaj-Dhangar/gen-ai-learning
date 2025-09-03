import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("salary_data.csv")
data_frame = pd.read_csv("salary_data.csv", sep="\t")

x = data_frame.drop('Salary',axis=1)
y=data_frame['Salary']
# planning to use equation y=mx+c where m is slope/coeficient and c is intercept
model = LinearRegression()

# fitting the x and y into the model meaning finding the m and c\
# once model get the value of m& c we can say model is trained
model.fit(x,y)

salary = model.predict(pd.DataFrame([[15]], columns= ['Experience']))

print("Salary of 15 years experience is:", salary[0])

# Model keep the values of m and c in coef_ and intercept_ variables
print("Coefficient (m):", model.coef_[0])# Slope or weight of the feature(column). how much y change when X changes by 1 unit
print("Intercept (c):", model.intercept_) # value of Y when x is 0

# we can put value of m&c in coef_ and intercept_ variables
m = model.coef_[0]
c = model.intercept_
salary= m*15 + c
print("Salary of 15 years experience is using formula:", salary)