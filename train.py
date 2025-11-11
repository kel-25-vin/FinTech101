import os
from stock_prediction import load_data, create_model, predict
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from parameters import *
from visualizations import *
import pandas as pd

# Create directories if they do not exist.
if not os.path.exists("models"):
    os.makedirs("models")

# Load the data.
data = load_data(ticker=TICKER, start_date=START_DATE, end_date=END_DATE,
                 test_size=TEST_SIZE,
                 store_locally=STORE_LOCALLY, load_locally=LOAD_LOCALLY,
                 local_data_path=LOCAL_DATA_PATH,
                 scale_data=SCALE_DATA,
                 split_by_date=SPLIT_BY_DATE,
                 history_days=HISTORY_DAYS, predict_days=PREDICT_DAYS,
                 feature_columns=FEATURE_COLUMNS)

# print(data["df"])
# print(data["X_train"])
# print(data["X_train"].shape)
# print(data["y_train"])
print(data["y_train"].shape)
# print(data["X_test"])
# print(data["X_test"].shape)
# print(data["y_test"])
print(data["y_test"].shape)
# print(data["test_df"])
# print(data["test_df"].shape)
# print(data["last_sequence"])
# print(data["last_sequence"].shape)

# Visualize the data using a multiple kinds of chart.
candlestick_visualization(data["df"], trading_days=7, feature_columns=FEATURE_COLUMNS)
boxplot_visualization(data["df"], price_columns=PRICE_COLUMNS)

# Construct the model.
model = create_model(HISTORY_DAYS, len(FEATURE_COLUMNS),
                     loss=LOSS_FUNCTION, units=UNITS, layer=LAYER, n_layers=N_LAYERS,
                     dropout=DROPOUT, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
print(model.summary())
print(model.output_shape)

# Callbacks.
checkpointer = ModelCheckpoint(filepath=f'models/{MODEL_NAME}.keras', save_best_only=True, verbose=1)

# Train the model.
TRAIN = False
if TRAIN:
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer], verbose=1)

# Predict the future stock prices.
best_model = load_model(f'models/{MODEL_NAME}.keras', compile=False)
predicted_price = predict(best_model, data)
LSTM_OUTPUT = pd.DataFrame(predicted_price, columns=[f'{LAYER.__name__}_Prediction'])[f'{LAYER.__name__}_Prediction']
print(LSTM_OUTPUT)
