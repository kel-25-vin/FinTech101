# File: stock_prediction.py
# Author: Kelvin Dang

# pip install scikit-learn
# pip install yfinance
# pip install pandas
# pip install numpy

import os
from collections import deque

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from parameters import *
from sentiments import vader_sentiment_analysis


def load_data(ticker, start_date, end_date, test_size=0.2,
              store_locally=False, load_locally=False, local_data_path=None,
              scale_data=True, split_by_date=True,
              history_days=50, predict_days=1, sentiment=False,
              news_data_path=None, classification=False,
              feature_columns=["Close", "High", "Low", "Open", "Volume"]):
    """
    Loads the stock data from Yahoo Finance, as well as:
        Custom start and end dates,
        Custom splitting,
        Optional local storage and loading,
        Optional scaling,
        Optional splitting by date,
        Custom history and prediction days.
    Returns a dictionary containing:
        "df": The original loaded DataFrame.
        "X_train", "X_test", "y_train", "y_test": The training and testing data.
        "test_df": The DataFrame for the testing data, indexed by date.
        "column_scaler": The dictionary of column scalers (if scale_data is True).
        "last_sequence": The last sequence of data, used for predicting future stock prices.
    Parameters:
        ticker (str): The stock ticker symbol.
        start_date (str): The start date for fetching stock data (format: "YYYY-MM-DD").
        end_date (str): The end date for fetching stock data (format: "YYYY-MM-DD").
        test_size (float): The proportion of the dataset to include in the test split (default is 0.25).
        store_locally (bool): Whether to store the downloaded data locally as a CSV file (default is False).
        load_locally (bool): Whether to load data from a local CSV file instead of downloading (default is False).
        local_data_path (str): The path to the local CSV file (required if load_locally is True).
        scale_data (bool): Whether to scale the feature columns using Min-Max scaling (default is True).
        split_by_date (bool): Whether to split the data into training and testing sets based on date (default is True).
        history_days (int): The number of past days to use for predicting the future (default is 50).
        predict_days (int): The number of days into the future to predict (default is 1).
        sentiment (bool): Whether to include sentiment analysis data as a feature (default is True).
        news_data_path (str): The path to the news data CSV file for sentiment analysis (required if sentiment is True).
        classification (bool): Whether the prediction task is lower/higher classification (default is False).
        feature_columns (list): The list of feature columns to use from the DataFrame (default includes common stock features).
    """
    # Include the sentiment data if provided.
    if sentiment:
        assert news_data_path is not None, 'News data path must be provided if sentiment analysis is enabled.'
        feature_columns.append('Sentiment')
        sentiment_series = vader_sentiment_analysis(news_data_path)
    # Check if the user wants to load data from a local CSV file.
    if load_locally:
        assert os.path.isfile(local_data_path), f"Local data file not found: {local_data_path}"
        # Load the data from the specified CSV file.
        df = pd.read_csv(local_data_path, index_col="Date", parse_dates=True)
        # Ensure that the data is valid, by checking the columns.
        for col in feature_columns:
            assert col in df.columns, f"Column \"{col}\" not found in local data."
    else:
        # Download stock data from Yahoo Finance.
        df = yf.download(ticker, start=start_date, end=end_date, multi_level_index=False)
    # Convert the data to a CSV file, and store it locally if requested.
    if store_locally:
        if not os.path.isdir("data"):
            os.mkdir("data")
        df.to_csv(STOCK_DATA_PATH)
    # If sentiment analysis is enabled, merge the sentiment scores into the main DataFrame.
    if sentiment:
        # Merge the sentiment scores into the main DataFrame.
        df = df.merge(sentiment_series.rename("Sentiment"), how="left",
                      left_index=True, right_index=True)
        # Fill any NaN sentiment values with 0 (neutral sentiment).
        df["Sentiment"].fillna(0, inplace=True)
    # Prepare the dictionary for things we want to return.
    result = {}
    # Include the DataFrame for returning.
    result["df"] = df.copy()
    # Scale the data if requested.
    if scale_data:
        column_scaler = {}
        # Scale the data from 0 to 1.
        for col in feature_columns:
            if col != 'Sentiment':  # Do not scale sentiment column
                scaler = MinMaxScaler()
                # Reshape the data to be a 2D array for the scaler.
                # The parameter -1 allows the function to determine the size of that dimension automatically.
                df[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))
                # Save the scaler for this column.
                column_scaler[col] = scaler
            else:
                # Because sentiment had already been normalized, use manual scaling.
                df['Sentiment'] = (df['Sentiment'] + 1) / 2  # Scale from [-1, 1] to [0, 1]
        # Save the column scalers for returning.
        result["column_scaler"] = column_scaler
    # Duplicate the "Date" index into a column for easier access later.
    df["Date"] = df.index
    # Create the target column by shifting "Close" upwards.
    df["Target"] = df["Close"].shift(-predict_days)
    if classification:
        df['Target'] = (df['Target'] >= df['Close']).astype(int)
    # The last 'predict_days' rows will have NaN target values, so get them before dropping.
    last_sequence = np.array(df[feature_columns].tail(predict_days))
    # Drop NaN values.
    df.dropna(inplace=True)

    # Create the sequences, which is the data collected over "history_days", used to predict the "Target".
    sequence_data = []
    sequences = deque(maxlen=history_days)
    for entry, target in zip(df[feature_columns + ["Date"]].values, df["Target"].values):
        sequences.append(entry)
        if len(sequences) == history_days:
            sequence_data.append([np.array(sequences), target])

    # Get the last sequence for prediction, by appending the last "history_days" sequence with the "predict_days" sequence.
    # For instance, if history_days=50 and predict_days=1, then we want the last 51 days.
    # This last_sequence will be used for predicting future stock prices that are not available in the dataset.
    last_sequence = list([s[:len(feature_columns)] for s in sequences]) + list(last_sequence)
    last_sequence = np.array(last_sequence).astype(np.float32)
    # Add this to the results.
    result["last_sequence"] = last_sequence

    # Construct the feature sequences and targets.
    X, y = [], []
    for seq, target in sequence_data:
        X.append(seq)
        y.append(target)
    # Convert to numpy arrays.
    X = np.array(X)
    y = np.array(y)

    if split_by_date:
        # Split the data into training and testing sets based on date, not randomly splitting.
        train_samples = int((1 - test_size) * len(X))
        result["X_train"] = X[:train_samples]
        result["y_train"] = y[:train_samples]
        result["X_test"] = X[train_samples:]
        result["y_test"] = y[train_samples:]
    else:
        # Randomly split the data into training and testing sets.
        result["X_train"], result["X_test"], result["y_train"], result["y_test"] = train_test_split(X, y,
                                                                                                    test_size=test_size)
    
    # Get the list of test dates.
    dates = result["X_test"][:, -1, -1]
    # Retrieve test features from the original DataFrame.
    result["test_df"] = result["df"].loc[dates]
    # Remove duplicated dates in the testing DataFrame.
    result["test_df"] = result["test_df"][~result["test_df"].index.duplicated(keep="first")]
    # Remove dates from the training and testing DataFrames, and convert to float32.
    result["X_train"] = result["X_train"][:, :, :len(feature_columns)].astype(np.float32)
    result["X_test"] = result["X_test"][:, :, :len(feature_columns)].astype(np.float32)
    # Reshape y to 2D array for Keras compatibility.
    result["y_train"] = result["y_train"].astype(np.float32).reshape(-1, 1)
    result["y_test"] = result["y_test"].astype(np.float32).reshape(-1, 1)
    
    # Return the result dictionary.
    return result


def create_model(sequence_length, n_features, classification=False, units=256, layer=LSTM, n_layers=2, dropout=0.3,
                 loss="mean_absolute_error", optimizer="rmsprop", bidirectional=False):
    """
    Creates and compiles a Sequential RNN model for stock price prediction.
    Parameters:
        classification (bool): Whether the prediction task is lower/higher classification (default is False).
        sequence_length (int): The length of the input sequences (number of past days).
        n_features (int): The number of features in the input data.
        classification (bool): Whether the prediction task is lower/higher classification (default is False).
        units (int): The number of units in each RNN layer (default is 256).
        layer (class): The type of RNN layer to use (default is LSTM). Options include LSTM, GRU, SimpleRNN.
        n_layers (int): The number of RNN layers to stack (default is 2).
        dropout (float): The dropout rate to apply after each RNN layer (default is 0.3).
        loss (str): The loss function to use for model compilation (default is "mean_absolute_error").
            Options include "mse", "mae", "huber".
        optimizer (str): The optimizer to use for model compilation (default is "rmsprop").
            Options include "adam", "rmsprop", "sgd".
        bidirectional (bool): Whether to use bidirectional RNN layers (default is False).
    """
    # Initialize the model.
    model = Sequential()
    for i in range(n_layers):
        if i == 0:
            # Create the input layer.
            if bidirectional:
                model.add(Bidirectional(layer(units, return_sequences=True),
                                        input_shape=(sequence_length, n_features)))
            else:
                model.add(layer(units, return_sequences=True,
                                input_shape=(sequence_length, n_features)))
        elif i == n_layers - 1:
            # Create the last layer.
            if bidirectional:
                model.add(Bidirectional(layer(units, return_sequences=False)))
            else:
                model.add(layer(units, return_sequences=False))
        else:
            # Create hidden layers.
            if bidirectional:
                model.add(Bidirectional(layer(units, return_sequences=True)))
            else:
                model.add(layer(units, return_sequences=True))
        # Add dropout after each layer.
        model.add(Dropout(dropout))
    # Create the output layer.
    if classification:
        model.add(Dense(1, activation="sigmoid"))
        # Compile the model.
        model.compile(loss="binary_crossentropy", metrics=['accuracy', 'f1_score'], optimizer=optimizer)
    else:
        model.add(Dense(1, activation="linear"))
        # Compile the model.
        model.compile(loss=loss, metrics=['mae'], optimizer=optimizer)
    return model


def predict(model, data, classification=False):
    """
    Using the trained model and the last sequence of data to predict the stock prices of the next "PREDICT_DAYS" days.
    Parameters:
        model (tf.keras.Model): The trained RNN model.
        data (dict): The dictionary returned by the load_data function, containing the last sequence and scalers.
    """
    # Initialize a list to store predicted prices.
    result = []
    # Retrieve the last sequence from the data dictionary.
    last_sequence = data["last_sequence"]
    # Loop over the sequence to predict multiple days into the future. (1 day forward to PREDICT_DAYS days forward.)
    for i in range(1, PREDICT_DAYS + 1):
        # Retrieve the sequence to predict that day (i.e., the sequence targeted at day i in the future).
        predict_sequence = last_sequence[i:(i + HISTORY_DAYS)]
        # Expand dimensions to match the model's expected input shape.
        predict_sequence = np.expand_dims(predict_sequence, axis=0)
        # Get the prediction.
        prediction = model.predict(predict_sequence)
        # Get the price, by inverting the scaling if previously scaled.
        if SCALE_DATA:
            predicted_price = data["column_scaler"]["Close"].inverse_transform(prediction)
        else:
            predicted_price = prediction
        # Append the predicted price to the list.
        result.append(predicted_price[0][0])
    # Convert the list to a numpy array.
    result = np.array(result)
    return result
