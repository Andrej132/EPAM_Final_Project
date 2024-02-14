import pandas as pd
import data_loader
import re
import joblib
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')

try:
    logging.basicConfig(level=logging.INFO)
    logging.info("Loading test data...")
    train_data, test_data = data_loader.load_data()
except FileNotFoundError:
    raise Exception("Data file not found")

try:
    logging.info("Preprocessing...")
    STOPWORDS = set(stopwords.words('english'))
    lemma = WordNetLemmatizer()


    def clean_review_with_lemmatizer(review_text):
        review_text = str(review_text).lower()
        review_text = re.sub('[^a-zA-Z]', ' ', review_text)
        review_text = word_tokenize(review_text)
        review_text = [word for word in review_text if word not in STOPWORDS]
        review_text = [lemma.lemmatize(word=w, pos='v') for w in review_text]
        review_text = [word for word in review_text if len(word) > 2]
        review_text = ' '.join(review_text)
        return review_text


    test_data['Clean_reviews_with_lemm'] = test_data['review'].apply(clean_review_with_lemmatizer)
    test_data.to_csv(r"vol/data/processed/new_test.csv", index=False)
except Exception as e:
    print("An error occurred during preprocessing: " + str(e))

try:
    logging.info("Loading the vectorizer")
    vectorizer = joblib.load('vol/outputs/model/tfidf_lemm.pkl')
    vectorizer_results = vectorizer.transform(test_data['Clean_reviews_with_lemm'])
    logging.info("Loading the model...")
    SVM_model = joblib.load('vol/outputs/model/svm_model.pkl')
    logging.info("Predicting...")
    predictions = SVM_model.predict(vectorizer_results)
    accuracy = accuracy_score(test_data['sentiment'], predictions)
    logging.info(f"Accuracy = {accuracy}")
    prediction_dataset = pd.DataFrame({'sentiment': test_data['sentiment'], 'predictions': predictions})
    logging.info("Saving prediction dataset...")
    prediction_dataset.to_csv('vol/outputs/predictions/prediction_dataset.csv', index=False)
    logging.info("Completed inference")

except Exception as e:
    print("An error occurred during training: " + str(e))