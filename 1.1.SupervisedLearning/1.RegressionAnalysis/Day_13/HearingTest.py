import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

df= pd.read_csv('hearing_test.csv', sep=',')

x = df.drop('test_result', axis = 1)
y = df['test_result']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2, random_state=32112312)
#building the model
model = LogisticRegression()
model.fit(x_train,y_train)  # training the model using training data

output = model.predict(pd.DataFrame([[69,25]], columns=['age', 'physical_score']))
print("Predicted output for age 30 and physical score 28 is: ", output)

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