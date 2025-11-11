import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk.sentiment.util
import pandas as pd
from parameters import COMBINED_AAPL_NEWS_PATH, FINBERT_MODEL_NAME


def vader_sentiment_analysis(path: str) -> pd.DataFrame:
    """Perform sentiment analysis on AAPL news data using Vader.
    Args:
        path (str): Path to the combined AAPL news CSV file.
    Returns:
        pd.DataFrame: DataFrame with Vader sentiment scores added.
    """
    # Input combined CSV AAPL data, and rename index to 'date'.
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.rename_axis('date', inplace=True)
    # Loaded DataFrame contains NaN values. Impute them as empty strings.
    df = df.fillna('')
    # Initialize Vader sentiment analyzer.
    vader = SentimentIntensityAnalyzer()
    # Perform sentiment analysis using Vader.
    df['vader_sentiment'] = df['title'].apply(vader.polarity_scores)
    # Calculate compound sentiment score from Vader results.
    df['vader_compound'] = df['vader_sentiment'].apply(lambda x: x['compound'])
    # Keep only compound sentiment score column.
    series = df['vader_compound'].copy()
    # -----
    # Filter out mondays to perform weekend averaging.
    mondays = series.index[series.index.weekday == 0]
    # Perform weekend rolling average to mondays.
    for monday in mondays:
        # Include Saturday, Sunday, and Monday for averaging.
        weekend_dates = [monday - pd.Timedelta(days=2),
                         monday - pd.Timedelta(days=1), monday]
        # Reindex in case of missing dates.
        vals = series.reindex(weekend_dates)
        # Calculate mean sentiment for the weekend, and assign to Monday.
        mean = vals.mean(skipna=True)
        if not pd.isna(mean):
            series.loc[monday] = mean
    # Remove weekend entries from the series.
    weekend_mask = series.index.weekday >= 5
    series = series[~weekend_mask]
    return series


if __name__ == "__main__":
    # Perform Vader sentiment analysis.
    aapl_sent = vader_sentiment_analysis(COMBINED_AAPL_NEWS_PATH)
    print(aapl_sent)
    print(aapl_sent.loc[aapl_sent < 0])