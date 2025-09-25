# panda help to handle data in tabular form
import pandas as pd
# matplotlib helps to visualize data
import matplotlib.pyplot as plt
# Numpy helps to handlle numerical , scientific, differential data
import numpy as np

# data frame is a 2-D data structure 
# like a table with rows and columns

data_frame = pd.read_csv("salary_data.csv", sep="\t")

print (data_frame.columns)

print(data_frame.describe())

correlation = data_frame['Experience'].corr(data_frame['Salary'])
covariance = np.cov(data_frame['Experience'], data_frame['Salary'])

print("Mean Salary:", data_frame['Salary'].mean())
print("Median Salary:", data_frame['Salary'].median())
print("Mode Salary:",data_frame['Salary'].mode()[0])

plt.scatter(data_frame['Experience'], data_frame['Salary'])
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()