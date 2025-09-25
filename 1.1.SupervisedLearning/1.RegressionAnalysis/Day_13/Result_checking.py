import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, f1_score

df= pd.read_csv('logistic_regression_results.csv', sep=',')

x = df.drop('Pass', axis= 1)
y = df['Pass']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=15)

model = LogisticRegression()
model.fit(x_train, y_train)

output = model.predict(pd.DataFrame([[4, 2]], columns=['Hours_Studied','Practice_Tests']))
print("predicted output:", output)


y_pred = model.predict(x_test)  # predicting the output using testing data
confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(confusion)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy score: ", accuracy)

precision = precision_score(y_test, y_pred, average='weighted') 
print("Precision:", precision)

f1Score = f1_score(y_test, y_pred, average='weighted')
print("F1 Score:", f1Score)
