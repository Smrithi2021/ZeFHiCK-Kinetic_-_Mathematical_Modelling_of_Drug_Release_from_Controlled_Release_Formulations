# train_model.py
import pandas as pd
import re
import string
import joblib
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

# Load your data
df = pd.read_excel("drugsCom_raw.xlsx")
df = df[df['condition'].isin(['Depression', 'High Blood Pressure', 'Diabetes, Type 2'])]
df['Sentiment'] = df['rating'].apply(lambda x: 'positive' if x >= 6 else 'negative')

# Text preprocessing
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(raw_review):
    text = BeautifulSoup(raw_review, 'html.parser').get_text()
    text = re.sub('[^a-zA-Z]', ' ', text)
    words = text.lower().split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return ' '.join(words)

df['clean_review'] = df['review'].apply(preprocess_text)

# Model pipeline
X = df['clean_review']
y = df['Sentiment']

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000)),
    ('svm', SVC(kernel='linear', probability=True))
])

pipeline.fit(X, y)

# Save model
joblib.dump(pipeline, 'model/svm_sentiment_model.pkl')
print("✅ Model saved at model/svm_sentiment_model.pkl")
