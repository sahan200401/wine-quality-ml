import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from preprocessing import preprocess_data
from sklearn.metrics import mean_squared_error,r2_score


df = pd.read_csv(r"C:\Users\jbasn\OneDrive\Documents\winde_data_set\Data\Raw\winequalityN.csv")

x , y = preprocess_data(df)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)

model = RandomForestRegressor(n_estimators = 100,random_state=42)
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

rmse = mean_squared_error(y_test,y_pred,squared = False)
r2 = r2_score(y_test,y_pred)

print("RMSE : ", rmse)
print("r^2 : ", r2)

joblib.dump(model,"wine_quality_model.pkl")
