import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os
import re

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('punkt')

# Define stopwords
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Preprocess the text by removing special characters, 
    converting to lowercase, tokenizing, and removing stopwords
    """
    if isinstance(text, str):
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Convert to lowercase
        text = text.lower()
        # Simple tokenization by splitting on whitespace
        # This avoids the NLTK punkt_tab issue
        tokens = text.split()
        # Remove stopwords
        tokens = [word for word in tokens if word not in stop_words]
        # Join tokens back into a string
        return ' '.join(tokens)
    else:
        # Return empty string if text is not a string (e.g., NaN)
        return ''

def train_model(dataset_path, output_dir='./models'):
    """
    Train a fake news detection model using TF-IDF and Naive Bayes
    """
    print("Loading data...")
    # Load the fake news dataset
    data = pd.read_csv(dataset_path)
    
    # Print dataset info for debugging
    print(f"Dataset columns: {data.columns.tolist()}")
    print(f"Sample data:\n{data.head(2)}")
    
    # Check if 'text' column exists
    if 'text' not in data.columns:
        # Try to find the appropriate text column
        potential_text_columns = ['content', 'article', 'news', 'text_content']
        found = False
        for col in potential_text_columns:
            if col in data.columns:
                print(f"Using '{col}' as the text column")
                data.rename(columns={col: 'text'}, inplace=True)
                found = True
                break
        
        if not found:
            print("Text column not found. Available columns:", data.columns)
            return
    
    # Handle missing values properly (avoiding chained assignment)
    data = data.copy()  # Create a copy to avoid the pandas warning
    data['text'] = data['text'].fillna('')
    
    print("Preprocessing text...")
    # Preprocess the text data
    data['processed_text'] = data['text'].apply(preprocess_text)
    
    # Create label based on dataset name or path
    # Since we're loading from 'Fake.csv', assume all entries are fake news (label=1)
    if 'Fake' in dataset_path:
        print("Dataset identified as containing fake news, assigning label=1")
        data['label'] = 1
    # If loading from 'True.csv' or 'Real.csv', assume all entries are real news (label=0)
    elif 'True' in dataset_path or 'Real' in dataset_path:
        print("Dataset identified as containing real news, assigning label=0")
        data['label'] = 0
    else:
        # If label can't be determined from filename, try using 'subject' field
        if 'subject' in data.columns:
            print("Using 'subject' column to derive labels")
            # Print unique subjects to help understand categories
            print(f"Unique subjects: {data['subject'].unique()}")
            
            # You might need to map specific subjects to fake/real
            # This is a placeholder - adjust based on actual subjects in your dataset
            fake_subjects = ['fake_news_category1', 'fake_news_category2']
            data['label'] = data['subject'].apply(lambda x: 1 if x in fake_subjects else 0)
        else:
            print("Cannot determine labels from dataset. Please specify a label column or provide separate datasets for fake/real news.")
            return
    
    # Prepare features and labels
    X = data['processed_text']
    y = data['label']
    
    print("Vectorizing text...")
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    
    print("Splitting data into train and test sets...")
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)
    
    print("Training model...")
    # Train a Naive Bayes classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    
    print("Evaluating model...")
    # Evaluate the model
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print("Saving model and vectorizer...")
    # Save the model and vectorizer
    joblib.dump(classifier, os.path.join(output_dir, 'fake_news_classifier.pkl'))
    joblib.dump(vectorizer, os.path.join(output_dir, 'tfidf_vectorizer.pkl'))
    
    print("Training complete!")
    return classifier, vectorizer

def predict_fake_news(text):
    """
    Predict whether a news article is fake or real.
    
    Parameters:
    -----------
    text : str
        The text of the news article
    
    Returns:
    --------
    prediction : int
        1 if the news is predicted to be fake, 0 if real
    probability : float
        Probability of the news being fake
    """
    try:
        # Load the trained model and vectorizer
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
        model_path = os.path.join(model_dir, 'fake_news_classifier.pkl')
        vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.pkl')
        
        # Check if model files exist
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            print(f"Model files not found at {model_path} or {vectorizer_path}")
            return None, None
        
        # Load model and vectorizer
        classifier = joblib.load(model_path)
        vectorizer = joblib.load(vectorizer_path)
        
        # Preprocess the input text
        processed_text = preprocess_text(text)
        
        # Transform text to TF-IDF features
        text_tfidf = vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = classifier.predict(text_tfidf)[0]
        
        # Get probability scores
        probabilities = classifier.predict_proba(text_tfidf)[0]
        fake_probability = probabilities[1] if prediction == 1 else probabilities[0]
        
        return int(prediction), float(fake_probability)
    
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        # Return default values in case of error
        return None, None

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        train_model(dataset_path)
    else:
        print("Please provide the dataset path as an argument.")
        print("Example: python train_model.py ./datasets/Fake.csv")