"""
TextMind: Personality Predictor - Web Application
Interactive Streamlit interface for MBTI personality type prediction.
Uses real Kaggle MBTI dataset with optimized caching.
"""

import streamlit as st
import pickle
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import re


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


def load_model_and_artifacts():
    """
    Load the trained model, vectorizer, and label encoder.
    Results are cached to avoid reloading on every interaction.
    
    Returns:
        tuple: (model, vectorizer, label_encoder) or (None, None, None) if files missing
    """
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        
        with open('vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        return model, vectorizer, label_encoder
    except FileNotFoundError:
        return None, None, None


@st.cache_resource
def get_model_artifacts():
    """
    Cached wrapper for loading model artifacts.
    This prevents reloading the model on every user interaction.
    """
    return load_model_and_artifacts()


def predict_personality(text, model, vectorizer, label_encoder):
    """
    Predict MBTI personality type from text.
    
    Args:
        text (str): User input text
        model: Trained LogisticRegression model
        vectorizer: TfidfVectorizer
        label_encoder: LabelEncoder for personality types
        
    Returns:
        tuple: (predicted_type, probabilities, confidence)
    """
    # Preprocess input
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text:
        return None, None, 0
    
    # Vectorize
    X = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = np.max(probabilities)
    
    # Decode label
    predicted_type = label_encoder.inverse_transform([prediction])[0]
    
    return predicted_type, probabilities, confidence


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="TextMind: Personality Predictor",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
        <style>
            .main-header {
                text-align: center;
                font-size: 2.5rem;
                font-weight: bold;
                color: #2E86AB;
                margin-bottom: 10px;
            }
            .main-subheader {
                text-align: center;
                font-size: 1.1rem;
                color: #666;
                margin-bottom: 30px;
            }
            .result-box {
                background-color: #E8F5E9;
                padding: 20px;
                border-radius: 10px;
                border-left: 5px solid #4CAF50;
            }
            .mbti-type {
                font-size: 3rem;
                font-weight: bold;
                color: #2E86AB;
                text-align: center;
                margin: 20px 0;
            }
            .confidence-score {
                font-size: 1.2rem;
                text-align: center;
                color: #666;
            }
        </style>
    """, unsafe_allow_html=True)
    
    # Sidebar - System Status
    with st.sidebar:
        st.markdown("### 🔧 System Status")
        st.divider()
        
        model, vectorizer, label_encoder = get_model_artifacts()
        
        if model is not None:
            st.success("✅ System Online")
            st.write("Model and vectorizer loaded successfully.")
            st.write(f"**Personality Types:** {len(label_encoder.classes_)}")
            with st.expander("View All Types"):
                types_list = ", ".join(sorted(label_encoder.classes_))
                st.write(types_list)
        else:
            st.warning("⚠️ Training Needed")
            st.write("Model files not found. Please run `train_model.py` first.")
            st.info("Run the following command to train the model:\n```bash\npython train_model.py\n```")
        
        st.divider()
        st.markdown("### ℹ️ About")
        st.write("**TextMind** predicts MBTI personality types from text using machine learning.")
        st.write("Trained on the official Kaggle MBTI dataset.")
        st.write("© 2025 TextMind Project")
    
    # Main Content
    st.markdown('<div class="main-header">🧠 TextMind: AI Personality Profiling</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subheader">Discover your personality type through artificial intelligence</div>', unsafe_allow_html=True)
    
    st.divider()
    
    # Check if model is available
    if model is None:
        st.error("❌ Model not available. Please train the model first.")
        return
    
    # Input Section
    st.markdown("### 📝 Input Your Text")
    st.write("Paste a sample of your writing, social media posts, or any text that represents your personality.")
    
    user_text = st.text_area(
        "Enter text for personality analysis:",
        height=200,
        placeholder="Type or paste your text here... (minimum 50 characters recommended)"
    )
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        analyze_button = st.button("🔍 Analyze Profile", use_container_width=True)
    
    st.divider()
    
    # Analysis Results
    if analyze_button:
        if not user_text or len(user_text.strip()) < 10:
            st.warning("⚠️ Please enter at least 10 characters of text for analysis.")
        else:
            # Perform prediction
            predicted_type, probabilities, confidence = predict_personality(
                user_text, model, vectorizer, label_encoder
            )
            
            if predicted_type is None:
                st.error("❌ Error: Unable to process the text. Please try again.")
            else:
                # Display main result
                st.markdown("### 🎯 Analysis Result")
                
                result_col1, result_col2 = st.columns([1, 1])
                
                with result_col1:
                    st.markdown(f'<div class="mbti-type">{predicted_type}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="confidence-score">Confidence: {confidence*100:.1f}%</div>', unsafe_allow_html=True)
                
                with result_col2:
                    st.info(f"""
                    **Personality Type:** {predicted_type}
                    
                    **Confidence Score:** {confidence*100:.2f}%
                    
                    This prediction is based on the linguistic patterns in your text.
                    """)
                
                # Top 3 predictions with probabilities
                st.markdown("### 📊 Top 3 Predicted Types")
                
                # Get top 3 probabilities
                top_3_indices = np.argsort(probabilities)[-3:][::-1]
                top_3_types = label_encoder.inverse_transform(top_3_indices)
                top_3_probs = probabilities[top_3_indices]
                
                # Create bar chart data
                chart_data = {
                    'Personality Type': top_3_types,
                    'Probability': top_3_probs * 100
                }
                
                st.bar_chart(
                    data=chart_data,
                    x='Personality Type',
                    y='Probability',
                    use_container_width=True
                )
                
                # Display detailed probabilities
                st.markdown("#### Detailed Breakdown")
                col1, col2, col3 = st.columns(3)
                
                for idx, (ptype, prob) in enumerate(zip(top_3_types, top_3_probs)):
                    with [col1, col2, col3][idx]:
                        st.metric(
                            label=f"#{idx+1}: {ptype}",
                            value=f"{prob*100:.1f}%"
                        )
                
                # Additional info
                st.markdown("---")
                with st.expander("📖 Understanding MBTI"):
                    st.write("""
                    The Myers-Briggs Type Indicator (MBTI) is a personality framework based on four dichotomies:
                    
                    - **E vs I:** Extraverted vs Introverted (how you direct energy)
                    - **S vs N:** Sensing vs Intuition (how you perceive information)
                    - **T vs F:** Thinking vs Feeling (how you make decisions)
                    - **J vs P:** Judging vs Perceiving (how you orient to the world)
                    """)


if __name__ == '__main__':
    main()
