"""
TextMind: Personality Predictor - Model Training Module
Trains a Logistic Regression model to predict MBTI personality types from text.
Uses the real Kaggle MBTI dataset via kagglehub.
"""

import pandas as pd
import numpy as np
import pickle
import os
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re
import warnings

warnings.filterwarnings('ignore')


def preprocess_text(text):
    """
    Clean and preprocess text data.
    
    Args:
        text (str): Raw text to preprocess
        
    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters and extra whitespace
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def download_kaggle_dataset():
    """
    Download the MBTI dataset from Kaggle using kagglehub.
    
    Returns:
        str: Path to the downloaded dataset folder
    """
    print("📥 Downloading MBTI dataset from Kaggle...")
    try:
        dataset_path = kagglehub.dataset_download("datasnaek/mbti-type")
        print(f"✅ Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("   Make sure you have Kaggle API configured.")
        print("   Run: kaggle auth login")
        return None


def load_kaggle_dataset(dataset_path):
    """
    Load the MBTI dataset from the Kaggle folder.
    
    Args:
        dataset_path (str): Path to the Kaggle dataset folder
        
    Returns:
        pd.DataFrame: DataFrame with 'type' and 'posts' columns
    """
    csv_file = os.path.join(dataset_path, 'mbti_1.csv')
    
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found at: {csv_file}")
        return None
    
    print(f"📖 Loading dataset from {csv_file}...")
    df = pd.read_csv(csv_file)
    print(f"✅ Loaded {len(df)} records from Kaggle dataset")
    
    return df


def train_model():
    """
    Main training function.
    Downloads from Kaggle, preprocesses, and trains the model.
    """
    # Download dataset from Kaggle
    dataset_path = download_kaggle_dataset()
    
    if dataset_path is None:
        print("❌ Failed to download dataset. Exiting.")
        return
    
    # Load dataset
    df = load_kaggle_dataset(dataset_path)
    
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    print("\n📊 Dataset Info:")
    print(f"   Total records: {len(df)}")
    print(f"   Unique personality types: {df['type'].nunique()}")
    print(f"   Personality types: {sorted(df['type'].unique())}")
    
    # Preprocess text
    print("\n🧹 Preprocessing text...")
    df['posts_cleaned'] = df['posts'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['posts_cleaned'].str.len() > 0]
    print(f"   Texts after cleaning: {len(df)}")
    
    # Vectorize text using TF-IDF
    print("\n🔢 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df['posts_cleaned'])
    
    print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"   Feature matrix shape: {X.shape}")
    
    # Encode labels
    print("\n🏷️  Encoding personality type labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['type'])
    
    print(f"   Classes: {label_encoder.classes_}")
    
    # Train Logistic Regression model
    print("\n🚀 Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
    model.fit(X, y)
    
    # Calculate training accuracy
    train_accuracy = model.score(X, y)
    print(f"   Training accuracy: {train_accuracy:.4f}")
    
    # Save model and vectorizer
    print("\n💾 Saving model and vectorizer...")
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("   ✅ Saved model.pkl")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("   ✅ Saved vectorizer.pkl")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("   ✅ Saved label_encoder.pkl")
    
    print("\n✨ Training complete! Model is ready for inference.")


if __name__ == '__main__':
    train_model()
