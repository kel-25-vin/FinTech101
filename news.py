from datetime import datetime, timedelta
import pandas as pd
import finnhub
from parameters import (KAGGLE_AAPL_NEWS_PATH, START_DATE, END_DATE,
                        FINNHUB_API_KEY, FINNHUB_AAPL_NEWS_PATH,
                        COMBINED_AAPL_NEWS_PATH)


def load_kaggle_aapl_data(start_date, end_date):
    """Load and preprocess Kaggle AAPL news dataset.
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: Preprocessed news data with date as index.
    """
    # Load Kaggle AAPL news dataset.
    df = pd.read_csv(KAGGLE_AAPL_NEWS_PATH, index_col='date',
                     parse_dates=True).sort_index()
    # Make sure the index contains only date, similar to the stock data.
    df.index = df.index.date
    df.index = pd.to_datetime(df.index)
    # Select data only after the specified start date.
    df = df.loc[start_date:]
    # Select only relevant columns (title, content).
    df = df[['title', 'content']]
    # Append a space to title and content to separate concatenated texts.
    df['title'] += ' '
    df['content'] += ' '
    # Group by date and aggregate (sum).
    df = df.groupby(pd.Grouper(freq='D')).sum()
    # Convert input date strings to datetime objects.
    start_date_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    # Create a complete date range from start to end date.
    all_dates = pd.date_range(start=start_date_dt, end=end_date_dt, freq='D')
    # Reindex the DataFrame to include all dates in the range.
    df = df.reindex(all_dates)
    df = df.rename_axis('date')
    # Fill missing and zero values: empty strings for text.
    df.fillna('', inplace=True)
    df.replace(0, '', inplace=True)
    # Drop the last row, as it exceeds the stock data.
    df.drop(df.index[-1], inplace=True)
    # Return the preprocessed Kaggle DataFrame.
    return df


def fetch_finnhub_data(start_date, end_date):
    """Fetch Finnhub AAPL news data and save to local CSV.
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    """
    # Initialize Finnhub client.
    client = finnhub.Client(api_key=FINNHUB_API_KEY)
    # Adjust end date not to be inclusive, similar to Yahoo Finance.
    end_date_dt = datetime.strptime(end_date, '%Y-%m-%d')
    end_date_dt -= timedelta(days=1)
    end_date = end_date_dt.strftime('%Y-%m-%d')
    # Fetch company news for AAPL from Finnhub.
    data = client.company_news('AAPL', _from=start_date, to=end_date)
    # Convert to DataFrame and save the fetched data to local CSV.
    data = pd.DataFrame(data)
    data.to_csv(FINNHUB_AAPL_NEWS_PATH, index=False)
    # CLI output confirmation.
    print(f"Fetched Finnhub data from {start_date} to {end_date}"
          f"and saved to {FINNHUB_AAPL_NEWS_PATH}.")
    return data


def load_finnhub_aapl_data():
    """Load and preprocess Finnhub AAPL news dataset.
    Args:
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
    Returns:
        pd.DataFrame: Preprocessed news data with date as index.
    """
    # Load Finnhub AAPL news dataset from local storage.
    df = pd.read_csv('data/finnhub_aapl_news.csv', index_col='datetime',
                     parse_dates=True).sort_index()
    # Drop irrelevant columns, retain only headline and summary.
    df.drop(columns=['category', 'id', 'image', 'related', 'source',
                     'url'], inplace=True)
    # Drop missing entries.
    df.dropna(inplace=True)
    # Convert index from UNIX timestamp to datetime, and retain only date.
    df.index = pd.to_datetime(df.index, unit='s').date
    df.index = pd.to_datetime(df.index)
    # Append a space to headline and summary to separate concatenated texts.
    df['headline'] += ' '
    df['summary'] += ' '
    # Group by date and aggregate by summing the text fields.
    df = df.groupby(pd.Grouper(freq='D')).sum()
    # Rename index and columns for integrity.
    df.rename_axis('date', inplace=True)
    df.rename(columns={'headline': 'title', 'summary': 'content'},
              inplace=True)
    # Return the preprocessed Finnhub DataFrame.
    return df


if __name__ == "__main__":
    kaggle_data = load_kaggle_aapl_data(START_DATE, END_DATE)
    print(kaggle_data)
    finnhub_data = load_finnhub_aapl_data()
    print(finnhub_data)
    news_data = kaggle_data.add(finnhub_data, fill_value='')
    print(news_data)
    news_data.to_csv(COMBINED_AAPL_NEWS_PATH)
