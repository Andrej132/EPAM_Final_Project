import os
import logging
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src.train.train import PROCESSED_DATA_DIR, MODELS_DIR, OUTPUTS_DIR

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, 'predictions')
if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)

logger.info("Loading processed test data...")
test_df = pd.read_csv(os.path.join(PROCESSED_DATA_DIR, 'test_processed.csv'))

logger.info("Loading models...")
model = joblib.load(os.path.join(MODELS_DIR, 'svm_model.pkl'))
vectorizer = joblib.load((os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl')))

logger.info("Vectorizing test data with TF-IDF...")
X_test_vectorized = vectorizer.transform(test_df['review'])

logger.info("Making predictions...")
predictions = model.predict(X_test_vectorized)
test_df['predicted_sentiment'] = predictions

logging.info("Evaluating...")
y_test = test_df['sentiment']
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
logger.info(f"Model accuracy: {accuracy:.4f}")
logger.info(f"Classification report:\n{report}")

logger.info("Saving predictions...")
test_df.to_csv(os.path.join(PREDICTIONS_DIR, 'predictions.csv'), index=False)
logger.info("Predictions saved successfully.")





