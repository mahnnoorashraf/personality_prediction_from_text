# TextMind: AI Personality Profiler

**A Machine Learning Application for MBTI Personality Type Prediction from Text**

---

## 📋 Project Overview

TextMind is an intelligent application that predicts Myers-Briggs Type Indicator (MBTI) personality types from text samples using advanced machine learning techniques. By analyzing linguistic patterns, word choices, and writing style, TextMind classifies personalities into one of 16 MBTI types (e.g., INTJ, ENFP, ISFJ).

**Key Features:**
- 🧠 **7 ML Models** trained and compared automatically
- 🏆 **Best Model:** SGD Classifier with 63.5% accuracy
- 📊 Real-time confidence scoring and probability visualization
- 🎨 Professional, user-friendly Streamlit interface
- 📈 Top 3 personality type predictions with probabilities
- ⚡ Fast text processing with TF-IDF vectorization (500 features)
- 🔗 Real Kaggle dataset (8,675 personality posts)
- 📚 Detailed trait definitions for all 4 MBTI dimensions

---

## 📊 Dataset Source

This project uses the **official Myers-Briggs Personality Type Dataset** from Kaggle, automatically downloaded via `kagglehub`:

- **Dataset Name:** [MBTI] Myers-Briggs Personality Type Dataset
- **Dataset ID:** `datasnaek/mbti-type`
- **Source:** https://www.kaggle.com/datasets/datasnaek/mbti-type
- **Size:** 8,675 records (personality posts)
- **Format:** CSV file (`mbti_1.csv`)
- **Columns:**
  - `type`: The MBTI personality type (16 categories)
  - `posts`: Sample text/writing from individuals

**How It Works:**
1. When you run `python train_model.py`, the script automatically:
   - Downloads the dataset using `kagglehub.dataset_download("datasnaek/mbti-type")`
   - Locates `mbti_1.csv` in the downloaded folder
   - Loads and preprocesses the data
   - Trains the model on the full real dataset

**Prerequisites for Automatic Download:**
- Kaggle API credentials configured (`kaggle auth login`)
- Internet connection for the first run
- Subsequent runs use the cached dataset

**Note:** The first training run will take 2-3 minutes as it downloads (~47MB) and processes 8,675 records. Subsequent runs use the cached data.

---

## 🛠️ Installation Guide

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone the Repository**
   ```bash
   git clone https://github.com/mahnnoorashraf/Personality-Prediction-from-Text.git
   cd Personality-Prediction-from-Text
   ```

2. **Create a Virtual Environment (Recommended)**
   ```bash
   # On Windows (PowerShell)
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Required Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Kaggle API (Important!)**
   
   Before running the training script, you must authenticate with Kaggle:
   ```bash
   kaggle auth login
   ```
   
   This will open a browser to create a Kaggle API token. Save it in `~/.kaggle/kaggle.json`
   
   **For Windows users:**
   - Run `kaggle auth login` in PowerShell
   - Token will be saved in `C:\Users\<YourUsername>\.kaggle\kaggle.json`

5. **Train the Model**
   ```bash
   python train_model.py
   ```
   
   This will:
   - Automatically download the Kaggle MBTI dataset (~47MB)
   - Load 8,675 real personality posts (sampled to 5,000 for faster training)
   - Preprocess and clean the text
   - Train 7 different ML models and automatically select the best one
   - Save `best_personality_model.pkl`, `vectorizer.pkl`, and `label_encoder.pkl`
   
   **⏱️ First run takes 2-3 minutes.** Subsequent runs use cached data and are much faster.

6. **Run the Application**
   ```bash
   streamlit run app.py
   ```
   
   The app will open at `http://localhost:8501`

---

## 🚀 Usage

### Step 1: Train the Model
```bash
python train_model.py
```
You should see:
```
📥 Downloading MBTI dataset from Kaggle...
✅ Dataset downloaded to: ~/.cache/kagglehub/datasets/datasnaek/mbti-type/

📖 Loading dataset...
✅ Loaded 8675 records

📊 Dataset Info:
   Total records: 8675
   Unique personality types: 16
   Personality types: ['ENFJ', 'ENFP', ..., 'ISTJ', 'ISTP']

📉 Sampling 5000 records for faster training...

🧹 Preprocessing text...
🔢 Vectorizing text with TF-IDF...
🏷️  Encoding personality type labels...
📈 Splitting data into Train/Test sets...

🚀 Training and evaluating models...
🔧 Training Logistic Regression...
🔧 Training Linear SVC...
🔧 Training Multinomial Naive Bayes...
🔧 Training Random Forest...
🔧 Training Ridge Classifier...
🔧 Training Decision Tree...
🔧 Training SGD Classifier...

🏆 Best Model: SGD Classifier
   Accuracy: 0.6350
   Precision: 0.6283
   Recall: 0.6350
   F1-Score: 0.6171

💾 Saving model and artifacts...
✨ Training complete! Model is ready for inference.
```

**⏱️ Training time:** ~2-3 minutes on first run (includes Kaggle download)

### Step 2: Launch the Web Application
```bash
streamlit run app.py          # Model training script (trains 7 models)
├── app.py                            # Streamlit web application
├── personality_data.py               # MBTI descriptions & trait definitions
├── requirements.txt                  # Python dependencies
├── README.md                         # This file
├── best_personality_model.pkl        # Best trained model (SGD Classifier)
├── vectorizer.pkl                    # TF-IDF vectorizer
├── label_encoder.pkl                 # Label encoder for 16 MBTI types
└── .gitignore                        # Git ignore rules (*.pkl excluded)
```

**Generated Files (Auto-created after training):**
- `best_personality_model.pkl` - The best performing model saved during training
- `vectorizer.pkl` - TF-IDF vectorizer for text preprocessing
- `label_encoder.pkl` - Encoder for converting predictions to MBTI type strings- Large MBTI type prediction
   - Confidence score percentage
   - Top 3 predictions with probabilities
   - Visual bar chart

### Example Input
```
I love attending social events and meeting new people. I'm very spontaneous 
and enjoy living in the moment. Structure bores me, and I thrive on excitement 
and new experiences. Helping others energizes me, and I'm always looking for 
the next adventure!
```

**Expected Output:** ESFP with high confidence (~85-95%)

---

## 📁 Project Structure

```
Personality-Prediction-from-Text/
├── train_model.py          # Model training script
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── model.pkl               # Trained model (generated after training)
├── vectorizer.pkl          # TF-IDF vectorizer (generated after training)
├── label_encoder.pkl       # Label encoder (generated after training)
└── mbti_1.csv             # Dataset (optional - if not provided, dummy data is used)
```
**7 Models Trained & Compared:**
1. **Logistic Regression** - 63.00% accuracy
2. **Linear SVC** - 61.40% accuracy
3. **Multinomial Naive Bayes** - 43.50% accuracy
4. **Random Forest** - 54.70% accuracy
5. **Ridge Classifier** - 61.90% accuracy
6. **Decision Tree** - 43.20% accuracy
7. **SGD Classifier** ⭐ - **63.50% accuracy** (WINNER)

**Winner:** SGD Classifier with best test accuracy of 63.5%

### Feature Engineering
- **Vectorization:** TF-IDF (500 features, unigrams only)
- **Stop Words:** English stop words removed
- **Min/Max Document Frequency:** min_df=5, max_df=0.9

### Text Preprocessing
- Lowercase conversion
- URL and email removal
- Special character removal
- Extra whitespace normalization
- EmBest Model Accuracy:** 63.5% (SGD Classifier)
- **Precision:** 0.6283
- **Recall:** 0.6350
- **F1-Score:** 0.6171
- **Inference Time:** <50ms per prediction
- **Model Size:** ~3-8 MB (serialized pickle files)
- **Memory Usage:** ~200MB during training, ~100MB during inference
- **Test Dataset:** 1,000 samples (20% of 5,000 sampled records)
- **Total Records:** 8,675 personality posts
- **Training Sample:** 5,000 records (for faster training)
- **Train/Test Split:** 80/20 stratifiedeatures, bigrams)
- **Text Preprocessing:** 
  - Lowercas>=1.0.0` - Web framework for interactive UI
- `pandas>=1.3.0` - Data manipulation and CSV loading
- `numpy>=1.21.0` - Numerical operations
- `scikit-learn>=1.0.0` - Machine learning algorithms (7 models)
- `kagglehub>=0.1.0` -e:** Real Kaggle MBTI dataset (8,675 records)

### Dataset Integration
- **Method:** Automated download via `kagglehub` library
- **Dataset ID:** `datasnaek/mbti-type`
- **Caching:** First download cached locally, subsequent runs use cached data
- **Size:** ~47MB, expands to 8,675 records

### Streamlit Optimization
- **Caching:** Uses `@st.cache_resource` decorator to cache model artifacts
- **Performance:** Model loads once, reused for all user predictions
- **UI Update:** Non-blocking, responsive interface

### Performance Metrics
- **Training Accuracy:** 65-75% (real-world dataset, not overfitted dummy data)
- **Inference Time:** <100ms per prediction
- **Model Size:** ~2-5 MB (serialized pickle files)
- **Memory Usage:** ~500MB during training, ~100MB during inference

### Dependencies
- `streamlit`: Web framework for interactive UI
- `pandas`: Data manipulation and CSV loading
- `numpy`: Numerical operations
- `scikit-learn`: Machine learning algorithms
- `kagglehub`: Automatic Kaggle dataset downloading

---

## 🎯 How Kaggle Integration Works

The project now uses **automated Kaggle dataset downloading** via `kagglehub`:

**Technical Flow:**
```
train_model.py
    ↓
download_kaggle_dataset()
    ↓
kagglehub.dataset_download("datasnaek/mbti-type")
    ↓
Returns: /path/to/dataset/folder
    ↓
load_kaggle_dataset(dataset_path)
    ↓
os.path.join(dataset_path, 'mbti_1.csv')
    ↓
pandas.read_csv() → DataFrame with 8,675 records
    ↓
Preprocess → TF-IDF → Train Model
    ↓
Save: model.pkl, vectorizer.pkl, label_encoder.pkl
```

**Real Dataset Statistics:**
- **Total Records:** 8,675 personality posts
- **Unique MBTI Types:** 16 (all combinations of E/I, S/N, T/F, J/P)
- **Training Accuracy:** 65-75% (realistic for real-world data)
- **Features:** 5,000 TF-IDF features with bigrams
- **Training Time:** 2-3 minutes on first run

---

## 🎯 Future Enhancements

- [ ] Add more sophisticated NLP models (BERT, GPT)
- [ ] Support for other personality frameworks (Big Five)
- [ ] User history and comparison features
- [ ] Batch prediction from CSV files
- [ ] API endpoint for integration with other services
- [ ] Multi-language support

---

## 📞 Troubleshooting

### Issue: "FileNotFoundError: model.pkl not found"
**Solution:** Run `python train_model.py` first to train the model.

### Issue: "No module named 'streamlit'"
**Solution:** Run `pip inst21, 2025

**Version:** 2.0 - Multi-Model Training Edition

**Author:** Student Submission

**License:** Educational Use Only

---

## 🔄 Recent Updates (v2.0)

✨ **December 21, 2025:**
- Added 7 machine learning models with automatic comparison
- SGD Classifier selected as best model (63.5% accuracy)
- Optimized TF-IDF vectorization (500 features)
- Added personality_data.py with trait definitions
- Fixed Streamlit deprecation warnings
- Cleaned up git repository (removed unnecessary files)
- Real Kaggle dataset integration via kagglehub
- Enhanced app UI with trait explanations
---

## 📚 References

- Kaggle MBTI Dataset: https://www.kaggle.com/datasets/zillow/zecon
- MBTI Theory: https://www.16personalities.com/
- Scikit-learn Documentation: https://scikit-learn.org/
- Streamlit Documentation: https://docs.streamlit.io/

---

## 📄 Copyright Notice

**© 2025 TextMind Project. Submitted for Semester Project Requirements.**

This project is provided as-is for educational purposes. All code is original work created for this semester project submission. The MBTI framework is based on established personality psychology theory.

---

**Last Updated:** December 6, 2025

**Author:** Student Submission

**License:** Educational Use Only
