import os
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from src.data_loader import load_data
from src.data_processing import preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR,'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')
MODELS_DIR = os.path.join(OUTPUTS_DIR, 'model')

if not os.path.exists(MODELS_DIR):
    os.makedirs(MODELS_DIR)

def main():
    logger.info("Loading raw training data...")
    train_df, test_df = load_data('train.csv', 'test.csv', RAW_DATA_DIR)

    logger.info("Preprocessing training data...")
    preprocessed_train_df, preprocessed_test_df = preprocess_data(train_df, test_df, PROCESSED_DATA_DIR)

    logger.info("Splitting data into training and validation sets...")
    X = preprocessed_train_df['review']
    y = preprocessed_train_df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Vectorizing text data...")
    vectorizer = TfidfVectorizer()
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)

    logger.info("Training SVM model...")
    model = SVC(C=1.0, kernel='rbf', random_state=42)
    model.fit(X_train_vectorized, y_train)

    logger.info("Evaluating model...")
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"Model accuracy: {accuracy:.4f}")
    logger.info("Generating classification report...")
    report = classification_report(y_test, y_pred)
    logger.info(f"Classification report:\n{report}")

    logger.info("Saving model and vectorizer...")
    joblib.dump(model, os.path.join(MODELS_DIR, 'svm_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'tfidf_vectorizer.pkl'))

if __name__ == "__main__":
    main()