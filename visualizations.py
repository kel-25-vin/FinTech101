import pandas as pd
from parameters import *
import mplfinance as mpf
import matplotlib.pyplot as plt
import seaborn as sns

def candlestick_visualization(data, trading_days=1, feature_columns=["Close", "High", "Low", "Open", "Volume"]):
    """
    Visualizes the stock data using a candlestick chart, with intervals defined by the trading_days parameter.
    Parameters:
        data (pd.DataFrame): The stock data DataFrame.
        trading_days (int): The number of trading days to group together for each candlestick (default is 1).
        feature_columns (list): The list of feature columns to use from the DataFrame (default includes common stock features).
    """
    # Ensure the data is in the correct format.
    for col in feature_columns:
        assert col in data.columns, f"Column \"{col}\" not found in data."
    # Ensure the trading_days parameter is an integer not less than 1.
    assert isinstance(trading_days, int) and trading_days >= 1, "\"trading_days\" must be an integer not less than 1."
    # Group the data into windows of "trading_days" days.
    group = (pd.Series(range(len(data)), index=data.index) // trading_days)
    data_modified = data.groupby(group).agg({
        "Close": "last",
        "High": "max",
        "Low": "min",
        "Open": "first",
        "Volume": "sum"
    })
    # Set the index to the last date in each group.
    data_modified.index = data.groupby(group).apply(lambda x: x.index[-1])
    # Plot the candlestick chart.
    mpf.plot(
        data_modified,
        type="candle",
        style="binance",
        title=f"{TICKER} - Candlestick Chart ({trading_days}-Day Intervals)",
        ylabel="Price",
        volume=True,
        ylabel_lower="Shares\nTraded",
    )

def boxplot_visualization(data, price_columns=["Close", "High", "Low", "Open"]):
    """
    Visualizes the stock data using a boxplot for the specified price columns.
    Parameters:
        data (pd.DataFrame): The stock data DataFrame.
        price_columns (list): The list of price columns to visualize (default includes common price features).
    """
    # Ensure the data is in the correct format.
    for col in FEATURE_COLUMNS:
        assert col in data.columns, f"Column \"{col}\" not found in data."
    # Plot the boxplot.
    sns.boxplot(data=data[price_columns], orient="h")
    # Show the plot.
    plt.show()