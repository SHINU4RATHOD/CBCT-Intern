# 1. Importing Required Libraries and Dataset
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.express as px

# Importing libraries for machine learning and model evaluation
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

# Importing SQL library to interact with databases
from sqlalchemy import create_engine
import joblib
import pickle

# Reading the dataset into a pandas DataFrame
df = pd.read_csv('C:/Users/SHINU RATHOD/Desktop/internship assignment/07_CipherByte Technologies/03_Time Series Forecasting/Alcohol_Sales.csv')

# Pushing dataset to MySQL and retrieving data from MySQL using Python (common real-world practice)
engine = create_engine('mysql+pymysql://root:1122@localhost/cipherbyte_internship')
# df.to_sql('AS_TSF', con=engine, if_exists='replace', chunksize=1000, index=False)

# Querying the table back to check data retrieval
sql = 'select * from AS_TSF;'
df = pd.read_sql_query(sql, engine) 

# Quick exploration of the dataset
df.head()  # Displaying first 5 records
df.sample(10)  # Checking random 10 samples for further inspection
df.tail()  # Displaying the last 5 records to get a sense of the data
df.dtypes  # Verifying the data types of all columns
df.shape  # Checking the shape of the dataset (number of rows and columns)

# Displaying detailed information about the dataset (useful for understanding the structure)
df.info()  
df.describe()  # Summarizing the dataset's statistical properties
df.columns  # Viewing column names for clarity

# Checking for missing values and duplicated records
df.isnull().sum()  # Summing up any null/missing values
df.isnull().sum().sum()  # Confirming there are no missing values
df.duplicated().sum()  # Confirming there are no duplicated records

######################## Data Preprocessing ########################
# Renaming columns to make them more intuitive
df.columns = ['date', 'sales']    

# Converting 'date' column to datetime format for time series analysis
df['date'] = pd.to_datetime(df['date'])   

# Setting the 'date' column as the DataFrame index for time series processing
df.set_index('date', inplace=True)    

# Confirming the changes in data structure after the preprocessing steps
df.info()

# Plotting the time series to visually inspect any trends or seasonality
plt.plot(df['sales'])
plt.title('Alcohol Sales Over Time')
plt.show()

######################## Seasonal Decomposition ########################
# Decomposing the time series into trend, seasonal, and residual components
from statsmodels.tsa.seasonal import seasonal_decompose
from pylab import rcParams
# Adjusting figure size for better readability
rcParams['figure.figsize'] = 18, 8
# Applying additive model for decomposition as sales data is growing over time
decomposition = seasonal_decompose(df['sales'], model='additive', period=12)
# Plotting the decomposed components (trend, seasonal, residual)
fig = decomposition.plot()
plt.show()

######################## Stationarity Test (ADF Test) ########################
# Performing the Augmented Dickey-Fuller test to check whether the time series is stationary
from statsmodels.tsa.stattools import adfuller
# ADF test on original data (pre-differencing)
result = adfuller(df['sales'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])
# ADF Statistic: 2.0374047259136963
# p-value: 0.9987196267088919
# If the p-value is greater than 0.05, the series is non-stationary
# Since p > 0.05, the data is non-stationary, and differencing is required

# Differencing the series to make it stationary
df['diff'] = df['sales'].diff(periods=12)

# Dropping any NaN values created by differencing
df_diff = df['diff'].dropna()

# Re-checking for stationarity after differencing
result = adfuller(df_diff)
print('ADF Statistic after differencing:', result[0])
print('p-value after differencing:', result[1])
# ADF Statistic after differencing: -3.3393107296695406
# p-value after differencing: 0.013210159306746523
# Now p < 0.05, confirming that the data is stationary after differencing

# Plotting the differenced series to visualize the transformed data
plt.plot(df_diff)
plt.title('Differenced Alcohol Sales (Seasonal)')
plt.show()

######################## Rolling Statistics ########################
# Calculating rolling mean and standard deviation to visually inspect stationarity
rolling_mean = df['sales'].rolling(window=12).mean()  # 12-period rolling mean
rolling_std = df['sales'].rolling(window=12).std()  # 12-period rolling standard deviation

# Plotting the original data alongside rolling mean and rolling standard deviation
plt.plot(df['sales'], label='Original')
plt.plot(rolling_mean, label='Rolling Mean')
plt.plot(rolling_std, label='Rolling Std')
plt.legend()
plt.show()     

######################## ACF and PACF Analysis ########################
# Plotting Auto-Correlation Function (ACF) to identify potential lags for AR component
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plot_acf(df['diff'].dropna(), lags=30)  # ACF for differenced data
plt.show()

# Plotting Partial Auto-Correlation Function (PACF) to understand lagged relationships
plot_pacf(df['diff'].dropna(), lags=30)  # PACF for differenced data
plt.show()

###################### Model Building: Fit the SARIMA Model ########################
# Importing SARIMAX model for time series forecasting
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Defining optimal SARIMA model parameters after analyzing ACF/PACF and seasonal decomposition
p, d, q = 0, 1, 1  # Non-seasonal order (chosen based on analysis)  # best values 2, 1, 2( # MAE: 607.6505195210842, # RMSE: 761.322324880756) for non-seasonal values
P, D, Q, s = 1, 1, 1, 12  # Seasonal order for yearly seasonality

# Fitting the SARIMA model on the differenced data
model = SARIMAX(df_diff, order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_result = model.fit()

# Displaying a summary of the model's parameters and performance
print(sarima_result.summary())

######################## Model Evaluation ########################
# Splitting data into training and testing sets for evaluation (80% training, 20% testing)
train_size = int(len(df) * 0.8)
train, test = df['sales'][:train_size], df['sales'][train_size:]

# Fitting the SARIMA model on the training data
model = SARIMAX(train, order=(p, d, q), seasonal_order=(P, D, Q, s))
sarima_result = model.fit()

# Forecasting for the test period
forecast = sarima_result.forecast(steps=len(test))

# Plotting actual vs forecasted values for comparison
plt.plot(test.index, test, label='Actual')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.legend()
plt.show()

# Calculating performance metrics: RMSE and MAE
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(test, forecast))
mae = mean_absolute_error(test, forecast)
print(f'RMSE: {rmse}') 
print(f'MAE: {mae}')  
# RMSE: 687.4588559016211
# MAE: 562.5119714626684


######################## Future Forecasting ########################
# Forecasting future values (next 12 months) using the SARIMA model
future_forecast = sarima_result.forecast(steps=12)

# Plotting the future forecast to visualize predicted sales
plt.plot(np.arange(len(df), len(df)+12), future_forecast, label='Future Forecast', color='green')
plt.legend()
plt.show()





########################### tech for to choose best pdq values
# 1. Grid Search with AIC/BIC Optimization
import itertools
# Define p, d, q ranges
p = d = q = range(0, 3)  # we can adjust the range to explore more values.

# Generate all combinations of p, d, q
pdq_combinations = list(itertools.product(p, d, q))

best_aic = float('inf')
best_pdq = None

# Perform grid search over p, d, q values
for pdq in pdq_combinations:
    try:
        # Fit ARIMA model
        model = SARIMAX(df['sales'], order=pdq)
        result = model.fit()
        
        # Store the best model based on AIC
        if result.aic < best_aic:
            best_aic = result.aic
            best_pdq = pdq
    except:
        continue

print(f'Best ARIMA model: order={best_pdq} with AIC={best_aic}')
# Best ARIMA model: order=(2, 1, 2) with AIC=5393.064750100672




# 2. Auto ARIMA (pmdarima)
import pmdarima as pm
# Auto ARIMA to find the best p, d, q values
model = pm.auto_arima(df['sales'], 
                      seasonal=False,  # we can set True if our data has seasonality
                      stepwise=True,   # Use stepwise search to find the best model
                      suppress_warnings=True,
                      trace=True)      # Show the search process
print(f'Best ARIMA model: {model.order}')
# model.summary()
# Best ARIMA model: (0, 1, 1)       AIC=5399.757
