import pmdarima as pm
from stock_prediction import load_data
from parameters import *
import pandas as pd

# Load the data.
data = load_data(
    ticker=TICKER, start_date=START_DATE, end_date=END_DATE,
    test_size=TEST_SIZE,
    store_locally=STORE_LOCALLY, load_locally=LOAD_LOCALLY,
    local_data_path=LOCAL_DATA_PATH,
    scale_data=SCALE_DATA,
    split_by_date=SPLIT_BY_DATE,
    history_days=HISTORY_DAYS, predict_days=PREDICT_DAYS,
    feature_columns=FEATURE_COLUMNS,
)

# Fitting a stepwise model to find the best parameters for SARIMA.
stepwise_fit = pm.auto_arima(
    data['df']['Close'], start_p=1, start_q=1, max_p=3, max_d=2, max_q=3, m=7,
    start_P=0, start_Q=0, max_P=3, max_D=3, max_Q=3, seasonal=True, trace=True,
    error_action='ignore',
    suppress_warnings=True,
    stepwise=True,
)
print(stepwise_fit.summary())

# Fitting the SARIMA model with the best parameters found.
SARIMA_Forecast = pd.DataFrame(stepwise_fit.predict(n_periods=30))
SARIMA_Forecast.columns = ['SARIMA_Forecast']
SARIMA_OUTPUT = SARIMA_Forecast.reset_index()['SARIMA_Forecast']
print(SARIMA_OUTPUT)