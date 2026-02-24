# ==============================
# 1. IMPORT NECESSARY LIBRARIES
# ==============================
import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Save models
import pickle
import joblib

# Download NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-eng')

# ==============================
# 2. LOAD AND EXPLORE DATASET
# ==============================

# Load the dataset
df = pd.read_csv('food.csv')

print("Dataset Shape:", df.shape)
print("\nDataset Columns:")
print(df.columns.tolist())
print("\nFirst few rows:")
print(df.head())
print("\nMissing values:")
print(df.isnull().sum())

# ==============================
# 3. PREPROCESS TEXT DATA
# ==============================

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    """
    Clean and preprocess text data
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words and len(word) > 2]
    
    return ' '.join(tokens)

# Apply preprocessing to reviewtext
df['cleaned_review'] = df['reviewtext'].apply(preprocess_text)

# ==============================
# 4. CREATE SENTIMENT LABELS
# ==============================

# Create sentiment labels based on reviewrating
def categorize_sentiment(rating):
    """
    Categorize sentiment based on rating:
    - Positive: rating >= 4
    - Neutral: 3 <= rating < 4
    - Negative: rating < 3
    """
    if pd.isna(rating):
        return 'neutral'
    elif rating >= 4:
        return 'positive'
    elif rating >= 3:
        return 'neutral'
    else:
        return 'negative'

df['sentiment'] = df['reviewrating'].apply(categorize_sentiment)

# Check sentiment distribution
print("\nSentiment Distribution:")
print(df['sentiment'].value_counts())

# Convert sentiment to numerical labels
sentiment_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
df['sentiment_label'] = df['sentiment'].map(sentiment_mapping)

# ==============================
# 5. PREPARE DATA FOR TRAINING
# ==============================

# Check for empty reviews
df = df[df['cleaned_review'].str.len() > 0]

# Split data
X = df['cleaned_review']
y = df['sentiment_label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")

# ==============================
# 6. FEATURE EXTRACTION (TF-IDF)
# ==============================

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

# Fit and transform training data
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

print(f"\nTF-IDF Features shape (train): {X_train_tfidf.shape}")
print(f"TF-IDF Features shape (test): {X_test_tfidf.shape}")

# ==============================
# 7. TRAIN SENTIMENT ANALYSIS MODELS
# ==============================

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Naive Bayes': MultinomialNB()
}

# Dictionary to store results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Training {model_name}...")
    
    # Train the model
    model.fit(X_train_tfidf, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_tfidf)
    y_pred_proba = model.predict_proba(X_test_tfidf)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    
    # Store results
    results[model_name] = {
        'model': model,
        'accuracy': accuracy,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }
    
    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print(f"\nClassification Report for {model_name}:")
    print(classification_report(y_test, y_pred, 
                                target_names=['negative', 'neutral', 'positive']))

# ==============================
# 8. SELECT BEST MODEL
# ==============================

# Find best model based on accuracy
best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_model_name]['model']

print(f"\n{'='*50}")
print(f"BEST MODEL: {best_model_name}")
print(f"Accuracy: {results[best_model_name]['accuracy']:.4f}")

# ==============================
# 9. SAVE MODELS AND VECTORIZER
# ==============================

# Create a dictionary to save all components
sentiment_pipeline = {
    'vectorizer': tfidf_vectorizer,
    'model': best_model,
    'sentiment_mapping': sentiment_mapping,
    'model_name': best_model_name
}

# Save the complete pipeline as pickle file
with open('food_sentiment_model.pkl', 'wb') as f:
    pickle.dump(sentiment_pipeline, f)

# Also save using joblib (better for large models)
joblib.dump(sentiment_pipeline, 'food_sentiment_model.joblib')

print("\nModels saved successfully!")
print("1. food_sentiment_model.pkl")
print("2. food_sentiment_model.joblib")

# ==============================
# 10. CREATE PREDICTION FUNCTION
# ==============================

def predict_sentiment(text, pipeline=None):
    """
    Predict sentiment for new text
    """
    if pipeline is None:
        with open('food_sentiment_model.pkl', 'rb') as f:
            pipeline = pickle.load(f)
    
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Transform using TF-IDF
    text_tfidf = pipeline['vectorizer'].transform([cleaned_text])
    
    # Predict
    prediction = pipeline['model'].predict(text_tfidf)[0]
    probabilities = pipeline['model'].predict_proba(text_tfidf)[0]
    
    # Map prediction to sentiment
    reverse_mapping = {v: k for k, v in pipeline['sentiment_mapping'].items()}
    sentiment = reverse_mapping[prediction]
    
    return {
        'text': text,
        'cleaned_text': cleaned_text,
        'sentiment': sentiment,
        'sentiment_label': prediction,
        'probabilities': probabilities,
        'confidence': max(probabilities)
    }

# Test the prediction function
print("\n{'='*50}")
print("TESTING PREDICTION FUNCTION:")
test_texts = [
    "The food was amazing and service was excellent!",
    "Average experience, nothing special",
    "Terrible food, worst experience ever"
]

for text in test_texts:
    result = predict_sentiment(text)
    print(f"\nText: {text}")
    print(f"Sentiment: {result['sentiment']} (Confidence: {result['confidence']:.2%})")

# ==============================
# 11. ADD PREDICTIONS TO DATASET AND SAVE
# ==============================

# Add predictions to original dataset
df['predicted_sentiment_label'] = best_model.predict(tfidf_vectorizer.transform(df['cleaned_review']))
df['predicted_sentiment'] = df['predicted_sentiment_label'].map(
    {v: k for k, v in sentiment_mapping.items()}
)

# Add prediction probabilities
probabilities = best_model.predict_proba(tfidf_vectorizer.transform(df['cleaned_review']))
df['negative_prob'] = probabilities[:, 0]
df['neutral_prob'] = probabilities[:, 1]
df['positive_prob'] = probabilities[:, 2]

# Save enriched dataset
output_columns = [
    'name of restaurant', 'location', 'starrating', 'priceINR', 
    'amenities_cuisine', 'customer_names', 'reviewrating', 
    'reviewtext', 'cleaned_review', 'sentiment', 'predicted_sentiment',
    'negative_prob', 'neutral_prob', 'positive_prob', 'triptype'
]

df[output_columns].to_csv('food_reviews_with_sentiment.csv', index=False)

print("\n{'='*50}")
print("DATASET WITH SENTIMENT SAVED:")
print("File: food_reviews_with_sentiment.csv")
print(f"Records: {len(df)}")

# ==============================
# 12. ADDITIONAL ANALYSIS AND VISUALIZATION
# ==============================

def analyze_sentiment_by_restaurant(df):
    """
    Analyze sentiment distribution by restaurant
    """
    analysis = df.groupby('name of restaurant').agg({
        'reviewrating': 'mean',
        'sentiment': lambda x: x.value_counts().to_dict(),
        'predicted_sentiment': lambda x: x.value_counts().to_dict()
    }).reset_index()
    
    return analysis

# Create restaurant sentiment analysis
restaurant_analysis = analyze_sentiment_by_restaurant(df)

# Save restaurant analysis
restaurant_analysis.to_csv('restaurant_sentiment_analysis.csv', index=False)

print("\n{'='*50}")
print("RESTAURANT SENTIMENT ANALYSIS SAVED:")
print("File: restaurant_sentiment_analysis.csv")

# ==============================
# 13. MODEL DEPLOYMENT HELPER FUNCTIONS
# ==============================

def load_sentiment_model(model_path='food_sentiment_model.pkl'):
    """
    Load the trained sentiment analysis model
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def batch_predict_sentiments(texts):
    """
    Predict sentiments for multiple texts
    """
    model = load_sentiment_model()
    predictions = []
    
    for text in texts:
        prediction = predict_sentiment(text, model)
        predictions.append(prediction)
    
    return pd.DataFrame(predictions)

# ==============================
# 14. SUMMARY REPORT
# ==============================

print("\n{'='*50}")
print("SENTIMENT ANALYSIS SUMMARY")
print("="*50)
print(f"Total Reviews Analyzed: {len(df)}")
print(f"Training Samples: {len(X_train)}")
print(f"Testing Samples: {len(X_test)}")
print(f"\nSentiment Distribution:")
print(df['sentiment'].value_counts())
print(f"\nBest Model: {best_model_name}")
print(f"Model Accuracy: {results[best_model_name]['accuracy']:.4f}")
print(f"\nFiles Created:")
print("1. food_sentiment_model.pkl - Main model file")
print("2. food_sentiment_model.joblib - Alternative model file")
print("3. food_reviews_with_sentiment.csv - Dataset with predictions")
print("4. restaurant_sentiment_analysis.csv - Restaurant-level analysis")

# ==============================
# 15. SAMPLE USAGE EXAMPLE
# ==============================

print("\n{'='*50}")
print("SAMPLE USAGE:")
print("="*50)

print("""
# Load the model
with open('food_sentiment_model.pkl', 'rb') as f:
    pipeline = pickle.load(f)

# Predict sentiment for new text
def predict_new_review(text):
    # Preprocess
    cleaned = preprocess_text(text)
    # Transform
    features = pipeline['vectorizer'].transform([cleaned])
    # Predict
    prediction = pipeline['model'].predict(features)[0]
    sentiment = list(pipeline['sentiment_mapping'].keys())[
        list(pipeline['sentiment_mapping'].values()).index(prediction)
    ]
    return sentiment

# Example
result = predict_new_review("The food was delicious!")
print(f"Predicted Sentiment: {result}")
""")