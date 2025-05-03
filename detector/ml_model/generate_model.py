"""
Quick script to generate model files for the fake news detector.
"""
import os
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re

# Download necessary NLTK data
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess the text by removing special characters,
    converting to lowercase, and removing stopwords
    """
    if not isinstance(text, str):
        return ""
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    # Simple tokenization by splitting on whitespace
    tokens = text.split()
    # Remove stopwords
    tokens = [word for word in tokens if word not in stop_words]
    # Join tokens back into a string
    return ' '.join(tokens)

def create_simple_model():
    """
    Create a simple model with synthetic data for demonstration
    """
    print("Creating synthetic dataset...")
    
    # Create synthetic data
    titles = [
        "Breaking News: Major Discovery",
        "Scientists Find Cure for Common Cold",
        "Politicians Argue Over Budget",
        "Sports Team Wins Championship",
        "Celebrity Announces New Project",
        "Economy Shows Signs of Growth",
        "Technology Company Releases New Product",
        "Weather Report Predicts Rain",
        "Local Community Hosts Event",
        "International Relations Improve"
    ]
    
    # Create fake versions of these titles
    fake_titles = [
        "SHOCKING: Government Hiding Major Discovery",
        "Scientists SECRETLY Found Miracle Cure They Don't Want You To Know",
        "CORRUPT Politicians STEALING From Budget",
        "Sports Team CHEATED To Win Championship",
        "Celebrity's DARK SECRET Revealed",
        "Economy COLLAPSING Despite Official Reports",
        "Technology Company SPYING on Customers",
        "Weather Manipulation EXPOSED",
        "Local Community Event DISASTER Coverup",
        "International Conspiracy REVEALED"
    ]
    
    # Create a simple dataset
    real_news = pd.DataFrame({
        'text': titles * 5,  # Repeat to get more samples
        'label': [0] * (len(titles) * 5)  # 0 for real news
    })
    
    fake_news = pd.DataFrame({
        'text': fake_titles * 5,  # Repeat to get more samples
        'label': [1] * (len(fake_titles) * 5)  # 1 for fake news
    })
    
    # Combine datasets
    data = pd.concat([real_news, fake_news]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Created dataset with {len(data)} samples")
    
    # Preprocess text
    data['processed_text'] = data['text'].apply(preprocess_text)
    
    # Prepare features and labels
    X = data['processed_text']
    y = data['label']
    
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf = vectorizer.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    # Train model
    print("Training model...")
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Save model and vectorizer
    model_path = os.path.join(model_dir, 'fake_news_classifier.pkl')
    vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
    
    print(f"Saving model to {model_path}")
    joblib.dump(classifier, model_path)
    
    print(f"Saving vectorizer to {vectorizer_path}")
    joblib.dump(vectorizer, vectorizer_path)
    
    print("Model files created successfully!")
    return model_path, vectorizer_path

if __name__ == "__main__":
    model_path, vectorizer_path = create_simple_model()
    print(f"Model saved at: {model_path}")
    print(f"Vectorizer saved at: {vectorizer_path}")