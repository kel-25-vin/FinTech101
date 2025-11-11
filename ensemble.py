import yfinance as yf
from train import LSTM_OUTPUT
from sarima import SARIMA_OUTPUT
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error

# Remember to config the HISTORY_DAYS and PREDICT_DAYS from the parameters.py to be consistent with SARIMA.
# E.g., 365 and 30.

# Download new stock data for the ensemble prediction (30 days).
new_data = yf.download("CBA.AX", start='2025-07-31', end='2025-09-11', multi_level_index=False)
new_close_prices = new_data.reset_index()['Close']

# Calculate the ensemble prediction by averaging LSTM and SARIMA outputs.
ensemble_output = (LSTM_OUTPUT + SARIMA_OUTPUT) / 2
ensemble_df = pd.DataFrame({
    'Ensemble_Prediction': ensemble_output,
    'True_Close_Price': new_close_prices.values,
})
print(ensemble_df)
print(mean_absolute_percentage_error(ensemble_df['True_Close_Price'], ensemble_df['Ensemble_Prediction']))