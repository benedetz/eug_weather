import numpy as np
import pandas as pd
import sklearn

data_path = "../eug_weather/data.csv"
data = pd.read_csv(data_path)

# Null Analysis by Brayden
def display_number_of_null(data):
    data_is_null = data.isnull().sum()
    data_is_null = data_is_null.to_frame(name="Amount Null")
    data_is_null["Percent Null"] = ((data_is_null["Amount Null"] / len(data)) * 100).round(2)
    
    # Sort by Percent Null in descending order
    data_is_null = data_is_null.sort_values(by="Percent Null", ascending=False)

    print("\nNumber of data points: ", np.array(data).shape[0], "\n\n")
    print(data_is_null)
    
display_number_of_null(data)

data['Observation_Date'] = pd.to_datetime(data['Observation_Date'])

data['year'] = data['Observation_Date'].dt.year
data['month'] = data['Observation_Date'].dt.month
data['day'] = data['Observation_Date'].dt.day
data = data.drop(columns='Observation_Date')
print(data.describe())

def normalize_data_not_date(data, columns_to_scale, columns_to_avoid_scaling):
    scaler = sklearn.preprocessing.StandardScaler() # initalizing a scaler for data normalization
    scaled_data = scaler.fit_transform(data[columns_to_scale])
    scaled_data = pd.DataFrame(scaled_data, columns=columns_to_scale, index=data.index)
    scaled_data = pd.concat([scaled_data, data[columns_to_avoid_scaling]], axis=1)
    return scaled_data

avg_wind_speed_reg_model = sklearn.linear_model.LinearRegression()
rf_model = sklearn.ensemble.RandomForestRegressor(n_estimators=100, random_state=42)

dropped_columns = [
    "Station_ID",
    "Station_Name",
    "Time_Fastest_Mile",
    "Time_Peak_Gust",
    "Percent_Sunshine",
    "Snowfall",
    "Snow_Depth",
    "Avg_Temperature",
    "Total_Sunshine",
    "Fastest_2Min_Wind_Direction",
    "Fastest_5Sec_Wind_Direction",
    "Snow_Water_Equivalent",
    "Fastest_2Min_Wind_Speed",
    "Fastest_5Sec_Wind_Speed",
    "Fog_IceFog_HeavyFog",
    "Heavy_Fog_Mist",
    "Thunderstorms",
    "Ice_Pellets",
    "Hail",
    "Glaze_Rime",
    "Dust_BlowingDust_VolcanicAsh",
    "Smoke_Haze",
    "Blowing_Drifting_Snow",
    "Tornado_Funnel_Cloud",
    "Damaging_Winds",
    "Blowing_Spray",
    "Drizzle",
    "Freezing_Drizzle",
    "Rain",
    "Freezing_Rain",
    "Snow",
    "Snow_IcePellets_OnGround",
    "Ground_Fog",
    "Ice_Fog",
    "Wave_Height_Specific_Period",
    "Significant_Wave_Height",
]



avg_wind_speed_data = data.drop(columns=dropped_columns) 
rows_with_null_targets = avg_wind_speed_data[avg_wind_speed_data['Avg_Wind_Speed'].isnull()] # pulling out our missing values
avg_wind_speed_data_cleaned = avg_wind_speed_data.dropna(subset=['Avg_Wind_Speed'])

training_avg_wind_speed_data, non_train_data = sklearn.model_selection.train_test_split(avg_wind_speed_data_cleaned, test_size=0.3, random_state=42)
validation_avg_wind_speed_data, test_avg_wind_speed_data = sklearn.model_selection.train_test_split(non_train_data, test_size=0.5, random_state=42)

columns_to_scale = ['Precipitation', 'Min_Temperature', "Max_Temperature"]
columns_to_avoid_scaling = ['year', 'month', 'day']

# training data
train_features = training_avg_wind_speed_data.drop(columns='Avg_Wind_Speed')
scaled_train_data = normalize_data_not_date(train_features, columns_to_scale, columns_to_avoid_scaling)
train_targets = training_avg_wind_speed_data['Avg_Wind_Speed']


# Validation data
validation_features = validation_avg_wind_speed_data.drop(columns='Avg_Wind_Speed')
scaled_validation_data = normalize_data_not_date(validation_features, columns_to_scale, columns_to_avoid_scaling)
validation_targets = validation_avg_wind_speed_data['Avg_Wind_Speed']

# Test data
test_features = test_avg_wind_speed_data.drop(columns='Avg_Wind_Speed')
scaled_test_data = normalize_data_not_date(test_features, columns_to_scale, columns_to_avoid_scaling)
test_targets = test_avg_wind_speed_data['Avg_Wind_Speed']

# inputs for prediction
prediction_targets = rows_with_null_targets.drop(columns='Avg_Wind_Speed')


avg_wind_speed_reg_model.fit(scaled_train_data, train_targets)

param_grid = {
    'n_estimators': [100, 200, 500],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'max_features': ['sqrt', 'log2']
}
#grid_search = sklearn.model_selection.GridSearchCV(estimator=rf_model, param_grid=param_grid, scoring='r2', cv=3, verbose=2)
#grid_search.fit(scaled_train_data, train_targets)
#print(f"Best Parameters: {rf_model.best_params_}")

rf_model.fit(scaled_train_data, train_targets)


validation_predictions_reg = avg_wind_speed_reg_model.predict(scaled_validation_data)
validation_predictions_rf = rf_model.predict(scaled_validation_data)
mse = sklearn.metrics.mean_squared_error(validation_targets, validation_predictions_reg)
r2_reg = sklearn.metrics.r2_score(validation_targets, validation_predictions_reg)
r2_rf = sklearn.metrics.r2_score(validation_targets, validation_predictions_rf)
print(f"R² Score Regression: {r2_reg}")
print(f"R² Score RandomForest: {r2_rf}")
print(f"MSE: {mse}")

boolean_variables = [ 
    "Fog_IceFog_HeavyFog",
    "Heavy_Fog_Mist",
    "Thunderstorms",
    "Ice_Pellets",
    "Hail",
    "Glaze_Rime",
    "Dust_BlowingDust_VolcanicAsh",
    "Smoke_Haze",
    "Blowing_Drifting_Snow",
    "Tornado_Funnel_Cloud",
    "Damaging_Winds",
    "Blowing_Spray",
    "Drizzle",
    "Freezing_Drizzle",
    "Rain",
    "Freezing_Rain",
    "Snow",
    "Snow_IcePellets_OnGround",
    "Ground_Fog",
    "Ice_Fog",
    "Wave_Height_Specific_Period",
    "Significant_Wave_Height"
]

boolean_filled_data = data
boolean_filled_data[boolean_variables] = boolean_filled_data[boolean_variables].fillna(0)
display_number_of_null(boolean_filled_data)

data_without_null_avg_wind_speed = data
data_without_null_avg_wind_speed = data_without_null_avg_wind_speed.dropna(subset=["Avg_Wind_Speed"])
display_number_of_null(data_without_null_avg_wind_speed)

cleaned_data = data_without_null_avg_wind_speed.drop(columns='Percent_Sunshine')

def rain_to_snow_conversion(data):
    # Ensure only temperatures ≤ 32°F are considered for snowfall
    data.loc[data['Min_Temperature'] > 32, "Snowfall"] = 0

    # Apply snowfall conversion ratios
    conditions = [
        (data['Min_Temperature'] >= 34) & (data['Min_Temperature'] < 45),
        (data['Min_Temperature'] >= 27) & (data['Min_Temperature'] < 34),
        (data['Min_Temperature'] >= 20) & (data['Min_Temperature'] < 27),
        (data['Min_Temperature'] >= 15) & (data['Min_Temperature'] < 20),
        (data['Min_Temperature'] >= 10) & (data['Min_Temperature'] < 15),
        (data['Min_Temperature'] >= 0) & (data['Min_Temperature'] < 10),
        (data['Min_Temperature'] >= -20) & (data['Min_Temperature'] < 0),
        (data['Min_Temperature'] < -20)
    ]
    conversion_factors = [0.1, 10, 15, 20, 30, 40, 50, 100]
    
    for condition, factor in zip(conditions, conversion_factors):
        data.loc[condition, "Snowfall"] = data['Precipitation'] * factor

    return data

cleaned_data = rain_to_snow_conversion(cleaned_data)
cleaned_data.loc[cleaned_data["Min_Temperature"] > 32, "Snowfall"] = cleaned_data.loc[cleaned_data["Min_Temperature"] > 32, "Snowfall"].fillna(0)

def compute_snow_depth(df, col='Snowfall', window=5, melt_factor=0.2):
    df['Snow_Depth'] = 0
    for i in range(1, window + 1):
        df['Snow_Depth'] += df[col].shift(i, fill_value=0) * ((1 - melt_factor) ** i)
    
    df['Snow_Depth'] += df[col]  # Add current day's snowfall
    return df

cleaned_data = compute_snow_depth(cleaned_data, window=1)

cleaned_data['Avg_Temperature'] = (cleaned_data['Min_Temperature'] + cleaned_data['Max_Temperature']) / 2

cleaned_data.drop(columns=['Snow_Water_Equivalent'], inplace=True)
cleaned_data.drop(columns=["Time_Fastest_Mile", 'Time_Peak_Gust', 'Total_Sunshine'], inplace=True)


display_number_of_null(cleaned_data)

fastest_reg_model = sklearn.linear_model.LinearRegression()

# pulling out the values
rows_with_null_targets_direction = cleaned_data[cleaned_data['Fastest_2Min_Wind_Direction'].isnull()] # pulling out our missing values
rows_with_null_targets_speed = cleaned_data[cleaned_data['Fastest_2Min_Wind_Speed'].isnull()] # pulling out our missing values

# Dropping values
cleaned_data.dropna(subset=['Fastest_2Min_Wind_Direction'], inplace=True)
cleaned_data.dropna(subset=['Fastest_2Min_Wind_Speed'], inplace=True)

# pulling out test and validation for Fastest_2Min_Wind_Direction
training_direction, non_train_data_direction = sklearn.model_selection.train_test_split(cleaned_data, test_size=0.3, random_state=42)
validation_direction, test_direction = sklearn.model_selection.train_test_split(non_train_data_direction, test_size=0.5, random_state=42)

columns_to_scale = ['Avg_Wind_Speed', 'Snow_Depth', 'Percipitation', ]



# pulling out test and validation for Fastest_2Min_Wind_Direction
training_speed, non_train_data_speed = sklearn.model_selection.train_test_split(cleaned_data, test_size=0.3, random_state=42)
validation_speed, test_direction = sklearn.model_selection.train_test_split(non_train_data_speed, test_size=0.5, random_state=42)


fastest_reg_model = sklearn.linear_model.LinearRegression()

# pulling out the values
rows_with_null_targets_direction = cleaned_data[cleaned_data['Fastest_5Sec_Wind_Direction'].isnull()] # pulling out our missing values
rows_with_null_targets_speed = cleaned_data[cleaned_data['Fastest_5Sec_Wind_Speed'].isnull()] # pulling out our missing values

# Dropping values
cleaned_data.dropna(subset=['Fastest_5Sec_Wind_Direction'], inplace=True)
cleaned_data.dropna(subset=['Fastest_5Sec_Wind_Speed'], inplace=True)

# pulling out test and validation for Fastest_2Min_Wind_Direction
training_direction, non_train_data_direction = sklearn.model_selection.train_test_split(cleaned_data, test_size=0.3, random_state=42)
validation_direction, test_direction = sklearn.model_selection.train_test_split(non_train_data_direction, test_size=0.5, random_state=42)

columns_to_scale = ['Avg_Wind_Speed', 'Snow_Depth', 'Percipitation', ]



# pulling out test and validation for Fastest_2Min_Wind_Direction
training_speed, non_train_data_speed = sklearn.model_selection.train_test_split(cleaned_data, test_size=0.3, random_state=42)
validation_speed, test_direction = sklearn.model_selection.train_test_split(non_train_data_speed, test_size=0.5, random_state=42)

columns_to_scale = ['Avg_Wind_Speed', 'Snow_Depth', 'Precipitation']  

display_number_of_null(cleaned_data)