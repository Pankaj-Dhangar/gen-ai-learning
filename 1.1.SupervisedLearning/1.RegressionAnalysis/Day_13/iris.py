import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

df= pd.read_csv('iris.csv', sep=',')

x = df.drop('species', axis = 1)
y = df['species']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12332112)

model = LogisticRegression()
model.fit(x_train,y_train) 

output = model.predict(pd.DataFrame([[5,3.4,1.5,0.2]], columns=['sepal_length','sepal_width','petal_length','petal_width']))
print("predicted output:", output)

y_pred = model.predict(x_test)  # predicting the output using testing data
confusion = confusion_matrix(y_test,y_pred)
print("Confusion Matrix:")
print(confusion)

accuracy = accuracy_score(y_test,y_pred)
print("Accuracy score: ")
print(accuracy)

precision = precision_score(y_test,y_pred, average= 'weighted' )
print("Precision score: ")
print(precision)

f1Score= f1_score(y_test,y_pred, average='weighted')
print("F1 score: ")
print(f1Score)