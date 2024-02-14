# EPAM_Final_Project

# DS Part

## EDA with conclusions

We can see that there are two columns, reviews and sentiment, which is the target column. There are 40000 rows in this dataset while there are no NULL values. We can also see that the training dataset is balanced because half of the dataset has a target value of 'positive', while the other half of the dataset has a target value of 'negative'. From the review analysis, we can conclude that the reviews are quite long, while the average word length is about 5. This may mean that there are a large number of 'stopwords' that are not of great importance to us.

## Description of feature engineering 

Convert the text to lowercase - All words changes into lower case to avoid the duplication. Two same words can be considered as 2 separate words if this step is not done.

Removing all irrelevant characters - Remove numbers and punctuation if they are not relevant to your analyses. This method can also help in case we have the words Hey and Hey!. These two words will be considered different by the model without this step.

Tokenization - Tokenization is the process of splitting the given text into smaller pieces called tokens.

Removing stopwords - “Stopwords” are the most common words in a language like “the”, “a”, “me”, “is”, “to”, “all”,. These words do not carry important meaning and are usually removed from texts.

Lemmatization - Lemmatization consists in doing things properly with the use of a vocabulary and morphological analysis of words, to return the base or dictionary form of a word, which is known as the lemma.

Stemming - Stemming usually refers to a crude process that chops off the ends of words in the hope of achieving this goal correctly most of the time, and often includes the removal of derivational units.

Removing the small words - After performing all required process in text processing there is some kind of noise is present in our corpus, so like that i am removing the words which have very short length.

### Vectorizers

CountVectorizer:

CountVectorizer converts a collection of text documents into a matrix of token counts.
It represents each document as a vector where each element corresponds to the count of a specific word in the document.
It considers the frequency of words in a document but does not take into account the importance of words in the entire corpus.
It is a simple and efficient method for text representation.
It is suitable for tasks where the frequency of words is important, such as text classification or clustering based on word occurrence.

TF-IDF Vectorizer:

TF-IDF (Term Frequency-Inverse Document Frequency) Vectorizer also converts text documents into a numerical matrix.
It represents each document as a vector where each element corresponds to the TF-IDF score of a specific word in the document.
TF-IDF takes into account both the frequency of words in a document (TF) and the rarity of words in the entire corpus (IDF).
It assigns higher weights to words that are more important in a document but less frequent in the corpus.
It is useful for tasks where the importance of words in a document and their rarity in the corpus are both considered, such as information retrieval, document similarity, or text summarization.

## Reasonings on model selection

Advantages of SVM for Sentiment Analysis:

1. Effective in handling high-dimensional data: SVM performs well in high-dimensional spaces, making it suitable for sentiment analysis tasks that involve a large number of features or words.

2. Robust against overfitting: SVM has a regularization parameter that helps prevent overfitting, which is important when working with limited labeled data for sentiment analysis.

3. Ability to handle non-linear relationships: SVM can use different kernel functions (e.g., linear, polynomial, radial basis function) to capture non-linear relationships between features, allowing it to model complex sentiment patterns in text data.

4. Interpretable results: SVM provides clear decision boundaries, making it easier to interpret and understand the sentiment classification results.

Disadvantages of SVM for Sentiment Analysis:

1. Computationally expensive: SVM can be computationally expensive, especially when dealing with large datasets or complex kernel functions. Training an SVM model on a large corpus of text data may require significant computational resources and time.

2. Sensitivity to parameter tuning: SVM performance can be sensitive to the choice of hyperparameters, such as the regularization parameter (C) and the kernel function. Finding the optimal hyperparameters may require extensive experimentation and cross-validation.

## Overall performance evaluation

I used 3 different models for this project: Naive Bayes, Support Vector Machine and Logistic Regression. I also tested two vectorizers: CountVectorizer and TFIDF vectorizer, and I noticed that TFIDF vectorizer gives slightly better results The following table shows the accuracy : 

Results with Count vectorizer

|              | Naive Bayes | SVM  | Logistic Regression |
|--------------|-------------|------|---------------------|
| Lemmatization|    88.125   | 89.1 |        89.5         |
| Stemming     |   87.7625   | 89.0 |        89.3         |

Results with TF-IDF vectorizer 

|              | Naive Bayes |   SVM    | Logistic Regression |
|--------------|-------------|----------|---------------------|
| Lemmatization|    88.45    |  90.425  |       88.7125       |
| Stemming     |   88.3625   | 90.3875 |       88.6875       |

I chose the SVM model on which I applied lemmatization in preprocessing, and for vectorization I used the TFIDF vectorizer, because that model gave the best accuracy. 
Accuracy on the inference dataset is 91%. 

## Potential business applications and value for business

1. Customer feedback analysis: Sentiment analysis can be used to analyze customer feedback from various sources such as social media, online reviews, surveys, and customer support interactions. By understanding customer sentiment, businesses can identify areas of improvement, address customer concerns, and enhance customer satisfaction.

2. Risk management and fraud detection: Sentiment analysis can be applied in risk management and fraud detection scenarios. By analyzing sentiment in customer communications, social media, or financial data, businesses can identify potential risks, detect fraudulent activities, and take appropriate actions to mitigate them.

3. Customer service and support: Sentiment analysis can be used to analyze customer sentiment in real-time during customer support interactions. By understanding customer emotions and sentiment, businesses can provide personalized and empathetic responses, improve customer service experiences, and increase customer loyalty.

4. Brand monitoring and reputation management: Sentiment analysis can help businesses monitor and analyze the sentiment around their brand, products, or services. By tracking online conversations and sentiment, businesses can identify potential issues, manage their reputation, and take proactive measures to address any negative sentiment.


# ML Part

## How to run

It is necessary that the docker daemon is running. This can be achieved by running a docker desktop application. Then it is necessary to open the command prompt and enter the following command to create the volume:

`docker volume create final_vol`

Then the next command in the same command prompt:

`docker run -v final_vol:/vol_data  -it --name temp-container --rm busybox`

After that we need folders in our volume. It is necessary to open a new command prompt window and position yourself in the Final_Project folder. Next commands are: 

`docker cp data temp-container:/vol_data`

and

`docker cp outputs temp-container:/vol_data`

Then it is necessary to position in the src/train file to create the docker image:

`docker build -t train .`

Now let's run training

`docker run -v final_vol:/app/vol train`

If it's completed, we can position in src/inference file to create the docker image for inference:

`docker build -t inference .`

Run the inference:

`docker run -v final_vol:/app/vol inference`

That's everything!