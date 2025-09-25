import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

df= pd.read_csv('iris.csv', sep=',')

x = df.drop('species', axis = 1)
y = df['species']

Encoder = LabelEncoder()

y_encoded= Encoder.fit_transform(y)
model = LinearRegression()
model.fit(x,y_encoded)
#print(df.corr())
output = model.predict(pd.DataFrame([[5,3.4,1.5,0.2]], columns=['sepal_length','sepal_width','petal_length','petal_width']))
#print(Encoder.inverse_transform(output))
print(output)