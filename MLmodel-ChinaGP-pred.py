import fastf1
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


cache_folder = "MLmodel-ChinaGP-pred"
if not os.path.exists(cache_folder):
    os.makedirs(cache_folder)
fastf1.Cache.enable_cache(cache_folder)  

race = fastf1.get_session(2024,"Chinese Grand Prix", "R")
race.load()

laps_2024 = race.laps[['Driver','LapTime','Sector1Time','Sector2Time','Sector3Time']].copy()
laps_2024.dropna(inplace= True)

for col in ['LapTime','Sector1Time','Sector2Time','Sector3Time']:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

race_data = race.results[['DriverNumber','BroadcastName','TeamName','Position','Points']]

avg_sector_time = laps_2024.groupby("Driver")[['LapTime (s)','Sector1Time (s)','Sector2Time (s)','Sector3Time (s)']].mean().reset_index()

Q2025 = pd.DataFrame({
    "Driver": ["Oscar Piastri","George Russell","Lando Norris","Max Verstappen","Lewis Hamilton",
               "Charles Leclerc","Yuki Tsunoda","Alex Albon"],
    "QualifyingTime (s)": [90.641 , 90.723, 90.793, 90.817 , 90.927 ,
                           91.027 , 91.638 , 91.706]
})
driver_mapping = {
    "Lando Norris": "NOR","Oscar Piastri":"PIA","Max Verstappen":"VER",
    "George Russell":"RUS","Yuki Tsunoda":"TSU","Alex Albon":"ALB",
    "Charles Leclerc":"LEC","Lewis Hamilton":"HAM"
}

Q2025["DriverCode"] = Q2025["Driver"].map(driver_mapping)

merged_data = Q2025.merge(avg_sector_time , left_on = "DriverCode",right_on="Driver", how= "left",suffixes=('', '_avg'))
merged_data_complete = merged_data.dropna(subset=["Sector1Time (s)" , "Sector2Time (s)", "Sector3Time (s)","LapTime (s)"])

X = merged_data_complete[["QualifyingTime (s)","Sector1Time (s)" , "Sector2Time (s)", "Sector3Time (s)"]]
y = merged_data_complete["LapTime (s)"]

from sklearn.impute import SimpleImputer
imputerx = SimpleImputer(strategy = 'mean')
X = pd.DataFrame(imputerx.fit_transform(merged_data[["QualifyingTime (s)","Sector1Time (s)" , "Sector2Time (s)", "Sector3Time (s)"]]),
                 columns = ["QualifyingTime (s)","Sector1Time (s)" , "Sector2Time (s)", "Sector3Time (s)"],
                 index = merged_data.index )

imputery = SimpleImputer(strategy='mean')
y = pd.Series(imputery.fit_transform(merged_data[["LapTime (s)"]]).ravel(),index = merged_data.index,name = "LapTime (s)")

if X.shape[0] == 0:
    raise ValueError("Dataset is empty after preprocessing.Check Data Sources!")

X_train,X_test,y_train, y_test = train_test_split(X,y ,test_size=0.2,random_state=39)
model = GradientBoostingRegressor(n_estimators = 100,learning_rate=0.1,random_state=39)
model.fit(X_train,y_train)

predicted_lap_times = model.predict(X)
Q2025["PredictedRaceTime (s)"] = predicted_lap_times
Q2025 = Q2025.sort_values(by="PredictedRaceTime (s)")

print("\n Predicted 2025 Chinese GP WINNER \n")
print(Q2025[["Driver","PredictedRaceTime (s)"]])

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test , y_pred )
print("Mean Absolute Error : ", mae)