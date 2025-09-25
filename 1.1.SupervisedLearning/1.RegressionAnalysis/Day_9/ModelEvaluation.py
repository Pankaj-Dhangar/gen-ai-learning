import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
 
df = pd.read_csv('Spending_Data.csv', sep='\t')
X = df.drop('Spendings', axis=1)
y = df['Spendings']
X_train, X_test, y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=1000)

model = LinearRegression()
model.fit(X_train,y_train)

train_score = model.score(X_train, y_train)
test_score = model.score(X_test,y_test)

print(f"Train R^2 Score :{train_score}")
print(f"Test R^2 Score :{test_score}")

y_pred = model.predict(X_test)

#.... Mean Abulate error (MAE)
mae = mean_absolute_error(y_test, y_pred)

#... Mean Square error (mse)
mse = mean_squared_error(y_test,y_pred)

#... root mean square error (RMSE)
rmse = np.sqrt(mse)

# ... mean absolute presentage error (mape)
mape = np.mean(np.abs((y_test - y_pred)/ y_test))*100

