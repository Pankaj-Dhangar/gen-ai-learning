import pandas as pd 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

df= pd.read_csv('Salary_Data copy.csv', sep=',')

X= df.drop('Salary', axis=1)
y= df['Salary']

Coloum_Transformer = ColumnTransformer(
    transformers=[
        ("onehot", OneHotEncoder(sparse_output=True,drop='first'),['Title'])
    ],
    remainder="passthrough" #keep other columns as it is 
)
transformed_values = Coloum_Transformer.fit_transform(X)

print(transformed_values)
print(Coloum_Transformer.get_feature_names_out())

transformed_features = pd.DataFrame(transformed_values, columns=Coloum_Transformer.get_feature_names_out())

model = LinearRegression()
model.fit(transformed_features,y)

# Prepare new data for prediction

data_to_predict = pd.DataFrame([["Project Manager", 2]], columns=["Title", "Experience"])
print(data_to_predict)
new_data_transformed = Coloum_Transformer.transform(data_to_predict)
print(new_data_transformed)

data_frame_to_predict = pd.DataFrame(new_data_transformed, columns=Coloum_Transformer.get_feature_names_out())

y_pred = model.predict(data_frame_to_predict)
print(y_pred)