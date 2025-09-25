import pandas as pd
from sklearn.linear_model import LogisticRegression

df= pd.read_csv('hearing_test.csv', sep=',')

x = df.drop('test_result', axis = 1)
y = df['test_result']

model = LogisticRegression()
model.fit(x,y)
#print(df.corr())
output = model.predict(pd.DataFrame([[30,28]], columns=['age', 'physical_score']))  
print(output)
