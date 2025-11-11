import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix,
                             classification_report)
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from stock_prediction import load_data, create_model
from parameters import *

print(f'{LAYER.__name__}')

# Load the data.
data = load_data(
    ticker=TICKER, start_date=START_DATE, end_date=END_DATE,
    test_size=TEST_SIZE,
    store_locally=STORE_LOCALLY, load_locally=LOAD_LOCALLY,
    local_data_path=LOCAL_DATA_PATH,
    scale_data=SCALE_DATA,
    split_by_date=SPLIT_BY_DATE,
    history_days=HISTORY_DAYS, predict_days=PREDICT_DAYS,
    sentiment=SENTIMENT, news_data_path=COMBINED_AAPL_NEWS_PATH,
    classification=CLASSIFICATION,
    feature_columns=FEATURE_COLUMNS,
)

print(data["df"])
print(data["X_train"])
print(data['test_df'])
print(data["y_train"].shape)
print(data["y_test"].shape)
print(data['last_sequence'])

model = create_model(HISTORY_DAYS, CLASSIFICATION_N_FEATURES, classification=CLASSIFICATION,
                     units=UNITS, layer=LAYER, n_layers=N_LAYERS, dropout=DROPOUT,
                     loss=CLASSIFICATION_LOSS_FUNCTION, optimizer=OPTIMIZER, bidirectional=BIDIRECTIONAL)
print(model.summary())
print(model.output_shape)

# Callbacks.
checkpointer = ModelCheckpoint(filepath=f'models/{CLASSIFICATION_MODEL_NAME}.keras', save_best_only=True, verbose=1)

# Train the model.
TRAIN = False
if TRAIN:
    history = model.fit(data["X_train"], data["y_train"],
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=(data["X_test"], data["y_test"]),
                        callbacks=[checkpointer], verbose=1)

# Retrieve the best model.
best_model = load_model(f'models/{CLASSIFICATION_MODEL_NAME}.keras', compile=False)
# Make predictions.
y_pred_train = (best_model.predict(data["X_train"]) >= 0.5).astype(int)
y_pred_test = (best_model.predict(data["X_test"]) >= 0.5).astype(int)

# Evaluations.
metrics = {'accuracy': accuracy_score(data['y_test'], y_pred_test),
           'precision': precision_score(data['y_test'], y_pred_test),
           'recall': recall_score(data['y_test'], y_pred_test),
           'f1_score': f1_score(data['y_test'], y_pred_test),
           'confusion_matrix': confusion_matrix(data['y_test'], y_pred_test),
           'classification_report': classification_report(data['y_test'],
                                                          y_pred_test,
                                                          output_dict=True)}
print('Evaluation Metrics of the Sentiment Classification Model:')
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print('Confusion Matrix:')
print(metrics['confusion_matrix'])
print('Classification Report:')
print(pd.DataFrame(metrics['classification_report']).transpose())

print(data['test_df'].shape)  # (271, 6)
print(data['test_df'][data['test_df']['Sentiment'] >= 0].shape)  # (259, 6)
