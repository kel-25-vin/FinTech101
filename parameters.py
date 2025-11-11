import os
from tensorflow.keras.layers import LSTM

# Stock data information.
TICKER = "AAPL"
START_DATE = "2020-01-01"
END_DATE = "2025-08-01"

# User options.
# Local storage and loading options.
STORE_LOCALLY = True
LOAD_LOCALLY = False
STOCK_DATA_PATH = os.path.join("data", f"{TICKER}_{START_DATE}_{END_DATE}.csv")
LOCAL_DATA_PATH = None
# Feature scaling option.
SCALE_DATA = True
# Data preprocessing options.
SPLIT_BY_DATE = True
HISTORY_DAYS = 50
PREDICT_DAYS = 1
FEATURE_COLUMNS = ["Close", "High", "Low", "Open", "Volume"]
TEST_SIZE = 0.2
# Sentiment analysis option.
SENTIMENT = True
# Classification option.
CLASSIFICATION = True
# Visualization options.
PRICE_COLUMNS = ["Close", "High", "Low", "Open"]

# Model options.
LOSS_FUNCTION = "huber"  # Options: "mse", "mae", "huber"
UNITS = 256
LAYER = LSTM  # Options: "LSTM", "GRU", "SimpleRNN"
N_LAYERS = 2
DROPOUT = 0.3
OPTIMIZER = "adam"  # Options: "adam", "rmsprop", "sgd"
BIDIRECTIONAL = False
MODEL_NAME = f'{TICKER}_{LAYER.__name__}_layers{N_LAYERS}_units{UNITS}_dropout{int(DROPOUT*100)}_{OPTIMIZER}_{LOSS_FUNCTION}'
CLASSIFICATION_MODEL_NAME = f'{MODEL_NAME}_classification'

# If classification.
CLASSIFICATION_N_FEATURES = len(FEATURE_COLUMNS) + (1 if SENTIMENT else 0)
CLASSIFICATION_LOSS_FUNCTION = 'binary_crossentropy'

# Training options.
BATCH_SIZE = 64
EPOCHS = 500

# API keys.
FINNHUB_API_KEY = 'd41kjshr01qo6qdh3sogd41kjshr01qo6qdh3sp0'
# Finnhub AAPL news dataset path.
FINNHUB_AAPL_NEWS_PATH = 'data/finnhub_aapl_news.csv'
# Kaggle AAPL news dataset path.
KAGGLE_AAPL_NEWS_PATH = 'data/apple_news_data.csv'
# Combined news dataset path.
COMBINED_AAPL_NEWS_PATH = 'data/combined_aapl_news.csv'

# FinBERT model name.
FINBERT_MODEL_NAME = 'ProsusAI/finbert'
