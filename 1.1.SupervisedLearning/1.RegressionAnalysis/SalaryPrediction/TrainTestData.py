import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('Spending_Data.csv', sep='\t')
X = df.drop('Spendings', axis=1)
y = df['Spendings']
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=1000)

# spread operation in java script || unpacking in python 
#print("X_train:\n", X_train.head(),"\n")
#print("X_test:\n", X_test.head(),"\n")
#print("y_train:\n", y_train.head(),"\n")
#print("y_train:\n", y_train.head(),"\n")
 
model = LinearRegression()
model.fit(X_train,y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test,y_test)
print(f"Train Score : {train_score}")
print(f"Test Score : {test_score}")