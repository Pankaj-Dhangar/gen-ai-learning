import pandas as pd
from sklearn.impute import SimpleImputer
df = pd.read_csv('salary_data.csv', sep='\t')
#mean(average), median(midal), constant, most_frequent
imputer = SimpleImputer(strategy='most_frequent')
df['Salary'] = imputer.fit_transform(df[['Salary']])
print(df)