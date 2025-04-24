import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import os

#nltk.download('punkt')
#nltk.download('wordnet')
#nltk.download('stopwords')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess_review(review):
    review = review.lower()
    review = re.sub(r'http\S+|www\S+|https\S+', '', review, flags=re.MULTILINE)
    review = re.sub(r'[^a-z\s]', '', review)
    tokens = word_tokenize(review)
    processed_tokens = [
        lemmatizer.lemmatize(token) for token in tokens if token not in stop_words
    ]

    return ' '.join(processed_tokens)


def preprocess_data(train_data, test_data, processed_data_dir):
    train_data['review'] = train_data['review'].apply(preprocess_review)
    test_data['review'] = test_data['review'].apply(preprocess_review)

    train_data.to_csv(os.path.join(processed_data_dir, 'train_processed.csv'), index=False)
    test_data.to_csv(os.path.join(processed_data_dir, 'test_processed.csv'), index=False)

    return train_data, test_data

