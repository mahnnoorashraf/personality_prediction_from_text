"""
TextMind: Model Training Module - Enhanced Version
Trains multiple models and selects the best one for MBTI prediction.
Uses real Kaggle MBTI dataset via kagglehub.
"""

import pandas as pd
import numpy as np
import pickle
import os
import re
import warnings
import kagglehub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    print("   (This may take a few minutes on first run)")
    try:
        dataset_path = kagglehub.dataset_download("datasnaek/mbti-type")
        print(f"✅ Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"❌ Error downloading dataset: {e}")
        print("\n   If you get an authentication error:")
        print("   1. Go to https://www.kaggle.com/settings/account")
        print("   2. Click 'Create New API Token' to download kaggle.json")
        print("   3. Place it in ~/.kaggle/kaggle.json")
        print("   4. Run this script again")
        return None


def load_dataset(dataset_path):
    """
    Load the MBTI dataset from the Kaggle download folder.
    
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
    print(f"✅ Loaded {len(df)} records")
    
    return df


def train_model():
    """
    Main training function.
    Downloads Kaggle dataset, trains multiple models, and selects the best performer.
    """
    # Download dataset from Kaggle
    dataset_path = download_kaggle_dataset()
    
    if dataset_path is None:
        print("❌ Failed to download dataset. Exiting.")
        return
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    if df is None:
        print("❌ Failed to load dataset. Exiting.")
        return
    
    print("\n📊 Dataset Info:")
    print(f"   Total records: {len(df)}")
    print(f"   Unique personality types: {df['type'].nunique()}")
    print(f"   Personality types: {sorted(df['type'].unique())}")
    
    # Sample data for faster processing if dataset is too large
    if len(df) > 5000:
        print(f"\n📉 Sampling {5000} records for faster training...")
        df = df.sample(n=5000, random_state=42)
        print(f"   Sampled records: {len(df)}")
    
    # Preprocess text
    print("\n🧹 Preprocessing text...")
    df['posts_cleaned'] = df['posts'].apply(preprocess_text)
    
    # Remove empty texts
    df = df[df['posts_cleaned'].str.len() > 0]
    print(f"   Texts after cleaning: {len(df)}")
    
    # Vectorize text using TF-IDF
    print("\n🔢 Vectorizing text with TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        min_df=5,
        max_df=0.9,
        ngram_range=(1, 1),
        lowercase=True
    )
    X = vectorizer.fit_transform(df['posts_cleaned'])
    
    print(f"   Vocabulary size: {len(vectorizer.get_feature_names_out())}")
    print(f"   Feature matrix shape: {X.shape}")
    
    # Encode labels
    print("\n🏷️  Encoding personality type labels...")
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['type'])
    
    print(f"   Classes: {label_encoder.classes_}")
    
    # Split data
    print("\n📈 Splitting data into Train/Test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training samples: {X_train.shape[0]}")
    print(f"   Testing samples: {X_test.shape[0]}")
    
    # Define models to train - 7 reliable models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42,
            multi_class='multinomial',
            solver='lbfgs'
        ),
        "Linear SVC": LinearSVC(
            max_iter=2000,
            random_state=42,
            dual='auto'
        ),
        "Multinomial Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=50,
            random_state=42,
            max_depth=15
        ),
        "Ridge Classifier": RidgeClassifier(
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=20,
            random_state=42,
            min_samples_split=10
        ),
        "SGD Classifier": SGDClassifier(
            loss='log_loss',
            max_iter=1000,
            random_state=42,
            n_jobs=1
        )
    }
    
    # Train and evaluate all models
    print("\n🚀 Training and evaluating models...")
    print("=" * 70)
    
    results = {}
    best_model_name = None
    best_accuracy = 0
    best_model = None
    
    for model_name, model in models.items():
        print(f"\n🔧 Training {model_name}...")
        
        try:
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
            
            # Store results
            results[model_name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            # Print metrics
            print(f"   ✅ {model_name}")
            print(f"      Accuracy:  {accuracy:.4f}")
            print(f"      Precision: {precision:.4f}")
            print(f"      Recall:    {recall:.4f}")
            print(f"      F1-Score:  {f1:.4f}")
            
            # Track best model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
                best_model = model
                
        except Exception as e:
            print(f"   ❌ Error training {model_name}: {str(e)}")
    
    print("\n" + "=" * 70)
    print(f"\n🏆 Best Model: {best_model_name}")
    print(f"   Accuracy: {best_accuracy:.4f}")
    print(f"   Precision: {results[best_model_name]['precision']:.4f}")
    print(f"   Recall: {results[best_model_name]['recall']:.4f}")
    print(f"   F1-Score: {results[best_model_name]['f1']:.4f}")
    
    # Save the best model and artifacts
    print("\n💾 Saving model and artifacts...")
    
    with open('best_personality_model.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("   ✅ Saved best_personality_model.pkl")
    
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("   ✅ Saved vectorizer.pkl")
    
    with open('label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    print("   ✅ Saved label_encoder.pkl")
    
    print("\n✨ Training complete! Model is ready for inference.")
    print(f"   Model: {best_model_name}")
    print(f"   Test Accuracy: {best_accuracy:.4f}")


if __name__ == '__main__':
    train_model()
