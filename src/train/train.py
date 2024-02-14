import data_loader
import re
import joblib
import logging
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
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
    logging.info("Loading train data...")
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


    train_data['Clean_reviews_with_lemm'] = train_data['review'].apply(clean_review_with_lemmatizer)
    train_data.to_csv(r"vol/data/processed/new_train.csv", index=False)
except Exception as e:
    print("An error occurred during preprocessing: " + str(e))

try:
    logging.info("Start training...")
    logging.info("Splitting dataset...")
    X_train_lemm, X_test_lemm, y_train, y_test = train_test_split(train_data['Clean_reviews_with_lemm'],
                                                                  train_data['sentiment'], test_size=0.2,
                                                                  random_state=42)
    logging.info("Vectorizing...")
    tfidf_vectorizer_lemm = TfidfVectorizer(ngram_range=(1, 2))
    X_train_lemm_tfidf = tfidf_vectorizer_lemm.fit_transform(X_train_lemm)
    X_test_lemm_tfidf = tfidf_vectorizer_lemm.transform(X_test_lemm)

    logging.info("Saving the vectorizer...")
    joblib.dump(tfidf_vectorizer_lemm, 'vol/outputs/model/tfidf_lemm.pkl')

    svm_model_lemm = SVC(C=1.0, kernel='linear')
    svm_model_lemm.fit(X_train_lemm_tfidf, y_train)

    svm_pred_lemm = svm_model_lemm.predict(X_test_lemm_tfidf)

    svm_accuracy_lemm1 = accuracy_score(y_test, svm_pred_lemm)
    logging.info(f"Accuracy = {svm_accuracy_lemm1}")

    logging.info("Saving the SVM model...")
    joblib.dump(svm_model_lemm, 'vol/outputs/model/svm_model.pkl')
except Exception as e:
    print("An error occurred during training: " + str(e))
