import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
hotels_df = pd.read_csv('hotels.csv')
pois_df = pd.read_csv('pois.csv')

print("Hotels data shape:", hotels_df.shape)
print("POIs data shape:", pois_df.shape)
print("\nHotels columns:", hotels_df.columns.tolist())
print("\nPOIs columns:", pois_df.columns.tolist())

# Check for missing values
print("\nMissing values in hotels data:")
print(hotels_df.isnull().sum())

print("\nMissing values in POIs data:")
print(pois_df.isnull().sum())

# Analyze hotel data
print("\nHotel data statistics:")
print(f"Number of unique hotels: {hotels_df['Hotel Name'].nunique()}")
print(f"Star Rating range: {hotels_df['StarRating'].min()} to {hotels_df['StarRating'].max()}")
print(f"Price range: ₹{hotels_df['PriceINR'].min()} to ₹{hotels_df['PriceINR'].max()}")

# Create aggregated features per hotel
hotel_aggregated = hotels_df.groupby('Hotel Name').agg({
    'StarRating': 'mean',
    'PriceINR': 'mean',
    'Recall@10': 'mean',
    'NDCG@10': 'mean',
    'rouge1': 'mean',
    'rouge2': 'mean',
    'rougeL': 'mean',
    'bleu': 'mean'
}).reset_index()

# Add count of reviews per hotel
review_counts = hotels_df['Hotel Name'].value_counts().reset_index()
review_counts.columns = ['Hotel Name', 'review_count']
hotel_aggregated = hotel_aggregated.merge(review_counts, on='Hotel Name')

# Create target variable for ranking (using NDCG as proxy for relevance)
hotel_aggregated['relevance_score'] = hotel_aggregated['NDCG@10'] * hotel_aggregated['StarRating'] / 5

print("\nAggregated hotel data shape:", hotel_aggregated.shape)

# Clean POI data
pois_df = pois_df.dropna(subset=['Place Name'])

# Create features for POIs
# Convert Entry fee to numeric
def parse_entry_fee(fee_str):
    if pd.isna(fee_str) or fee_str == 'Free':
        return 0
    try:
        # Extract numeric value
        if '₹' in str(fee_str):
            return float(str(fee_str).split('₹')[1].split()[0])
        return float(fee_str)
    except:
        return 0

pois_df['Entry_fee_numeric'] = pois_df['Entry fee'].apply(parse_entry_fee)

# Extract distance as numeric
pois_df['Distance_numeric'] = pois_df['Distance(KM)'].str.extract('(\d+)').astype(float)

# Create categorical features
category_encoder = LabelEncoder()
pois_df['Category_encoded'] = category_encoder.fit_transform(pois_df['Category'])

# Create target variable for POIs (based on distance and popularity)
# Assuming closer and free POIs are more relevant
pois_df['poi_relevance'] = 1 / (pois_df['Distance_numeric'] + 1) + (pois_df['Entry_fee_numeric'] == 0).astype(float) * 0.5

# Hotel ranking features
hotel_features = [
    'StarRating', 'PriceINR', 'review_count',
    'rouge1', 'rouge2', 'rougeL', 'bleu',
    'Recall@10', 'NDCG@10'
]

X_hotels = hotel_aggregated[hotel_features]
y_hotels = hotel_aggregated['relevance_score']

# Normalize features
scaler_hotels = StandardScaler()
X_hotels_scaled = scaler_hotels.fit_transform(X_hotels)

# Split data
X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_hotels_scaled, y_hotels, test_size=0.2, random_state=42
)

# For POI ranking
poi_features = [
    'Category_encoded', 'Entry_fee_numeric', 'Distance_numeric'
]

X_pois = pois_df[poi_features].fillna(0)
y_pois = pois_df['poi_relevance']

# Normalize POI features
scaler_pois = StandardScaler()
X_pois_scaled = scaler_pois.fit_transform(X_pois)

# Split POI data
X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(
    X_pois_scaled, y_pois, test_size=0.2, random_state=42
)

def train_xgboost_ranking(X_train, X_test, y_train, y_test, model_name="hotel"):
    """
    Train XGBoost ranking model with hyperparameter tuning
    """
    print(f"\nTraining XGBoost Ranking Model for {model_name}...")
    
    # For ranking, we need to create query groups
    # Since each hotel/POI is independent, we'll create one query group per instance
    # But for XGBRanker, we need to specify group sizes
    train_groups = [len(X_train)]
    test_groups = [len(X_test)]
    
    # Create and train model with simplified parameters for faster execution
    xgb_model = xgb.XGBRanker(
        objective='rank:pairwise',
        random_state=42,
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        verbosity=0
    )
    
    # Fit the model
    xgb_model.fit(X_train, y_train, group=train_groups)
    
    # Make predictions
    y_pred_train = xgb_model.predict(X_train)
    y_pred_test = xgb_model.predict(X_test)
    
    # Calculate MAP (Mean Average Precision) - simplified version
    # We'll use average_precision_score for binary classification
    relevance_threshold = np.median(y_train)
    y_train_binary = (y_train >= relevance_threshold).astype(int)
    y_test_binary = (y_test >= relevance_threshold).astype(int)
    
    # For MAP, we need to sort predictions and calculate precision at each rank
    def calculate_map(y_true, y_pred):
        # Sort by predicted score in descending order
        sorted_indices = np.argsort(y_pred)[::-1]
        y_true_sorted = y_true[sorted_indices]
        
        # Calculate precision at each position
        precisions = []
        relevant_count = 0
        
        for i, is_relevant in enumerate(y_true_sorted, 1):
            if is_relevant:
                relevant_count += 1
                precisions.append(relevant_count / i)
        
        if len(precisions) == 0:
            return 0
        
        # Average precision is the mean of precisions at positions where item is relevant
        return np.mean(precisions)
    
    # Calculate MRR (Mean Reciprocal Rank)
    def calculate_mrr(y_true, y_pred):
        sorted_indices = np.argsort(y_pred)[::-1]
        for i, idx in enumerate(sorted_indices):
            if y_true[idx] == 1:
                return 1.0 / (i + 1)
        return 0
    
    map_train = calculate_map(y_train_binary.values, y_pred_train)
    map_test = calculate_map(y_test_binary.values, y_pred_test)
    
    mrr_train = calculate_mrr(y_train_binary.values, y_pred_train)
    mrr_test = calculate_mrr(y_test_binary.values, y_pred_test)
    
    print(f"\n{model_name.upper()} Model Results:")
    print(f"Train MAP: {map_train:.4f}")
    print(f"Test MAP: {map_test:.4f}")
    print(f"Train MRR: {mrr_train:.4f}")
    print(f"Test MRR: {mrr_test:.4f}")
    
    return xgb_model, map_test, mrr_test

# Train hotel ranking model
print("\n" + "="*50)
print("TRAINING HOTEL RANKING MODEL")
print("="*50)
hotel_model, hotel_map, hotel_mrr = train_xgboost_ranking(
    X_train_h, X_test_h, y_train_h, y_test_h, "hotel"
)

# Train POI ranking model
print("\n" + "="*50)
print("TRAINING POI RANKING MODEL")
print("="*50)
poi_model, poi_map, poi_mrr = train_xgboost_ranking(
    X_train_p, X_test_p, y_train_p, y_test_p, "poi"
)

# Save models
with open('hotel_ranking_model.pkl', 'wb') as f:
    pickle.dump({
        'model': hotel_model,
        'scaler': scaler_hotels,
        'features': hotel_features,
        'metrics': {'MAP': hotel_map, 'MRR': hotel_mrr}
    }, f)

with open('poi_ranking_model.pkl', 'wb') as f:
    pickle.dump({
        'model': poi_model,
        'scaler': scaler_pois,
        'features': poi_features,
        'metrics': {'MAP': poi_map, 'MRR': poi_mrr}
    }, f)

print("\nModels saved successfully!")
print("✓ Hotel model saved as: hotel_ranking_model.pkl")
print("✓ POI model saved as: poi_ranking_model.pkl")

# Load and evaluate baseline vs XGBoost
print("\n" + "="*50)
print("RANKING IMPROVEMENT ANALYSIS")
print("="*50)

# Baseline ranking (by StarRating for hotels, by Distance for POIs)
baseline_hotel_ranking = hotel_aggregated.sort_values('StarRating', ascending=False)
baseline_poi_ranking = pois_df.sort_values('Distance_numeric', ascending=True)

# XGBoost ranking predictions
hotel_aggregated['xgb_score'] = hotel_model.predict(scaler_hotels.transform(hotel_aggregated[hotel_features]))
xgb_hotel_ranking = hotel_aggregated.sort_values('xgb_score', ascending=False)

pois_df['xgb_score'] = poi_model.predict(scaler_pois.transform(pois_df[poi_features].fillna(0)))
xgb_poi_ranking = pois_df.sort_values('xgb_score', ascending=False)

# Calculate metrics for baseline
def calculate_ranking_metrics(df, relevance_col='relevance_score', ranking_col=None):
    if ranking_col is None:
        # Use index as ranking
        ranked_relevance = df[relevance_col].values
    else:
        # Sort by ranking column
        ranked_relevance = df.sort_values(ranking_col, ascending=False)[relevance_col].values
    
    # Convert to binary relevance (top 50% as relevant)
    threshold = np.median(ranked_relevance)
    binary_relevance = (ranked_relevance >= threshold).astype(int)
    
    # Calculate MAP
    def calculate_map_from_array(y_true):
        precisions = []
        relevant_count = 0
        
        for i, is_relevant in enumerate(y_true, 1):
            if is_relevant:
                relevant_count += 1
                precisions.append(relevant_count / i)
        
        if len(precisions) == 0:
            return 0
        return np.mean(precisions)
    
    # Calculate MRR
    def calculate_mrr_from_array(y_true):
        for i, rel in enumerate(y_true, 1):
            if rel == 1:
                return 1.0 / i
        return 0
    
    map_score = calculate_map_from_array(binary_relevance)
    mrr_score = calculate_mrr_from_array(binary_relevance)
    
    return map_score, mrr_score

# Calculate baseline metrics
baseline_hotel_map, baseline_hotel_mrr = calculate_ranking_metrics(
    hotel_aggregated, 'relevance_score', 'StarRating'
)
baseline_poi_map, baseline_poi_mrr = calculate_ranking_metrics(
    pois_df, 'poi_relevance', 'Distance_numeric'
)

# Calculate XGBoost metrics
xgb_hotel_map, xgb_hotel_mrr = calculate_ranking_metrics(
    hotel_aggregated, 'relevance_score', 'xgb_score'
)
xgb_poi_map, xgb_poi_mrr = calculate_ranking_metrics(
    pois_df, 'poi_relevance', 'xgb_score'
)

print("\nHOTEL RANKING COMPARISON:")
print(f"{'Metric':<15} {'Baseline':<10} {'XGBoost':<10} {'Improvement':<10}")
print(f"{'-'*45}")
print(f"{'MAP':<15} {baseline_hotel_map:.4f}{'':<4} {xgb_hotel_map:.4f}{'':<4} {((xgb_hotel_map-baseline_hotel_map)/baseline_hotel_map*100):.1f}%")
print(f"{'MRR':<15} {baseline_hotel_mrr:.4f}{'':<4} {xgb_hotel_mrr:.4f}{'':<4} {((xgb_hotel_mrr-baseline_hotel_mrr)/baseline_hotel_mrr*100):.1f}%")

print("\nPOI RANKING COMPARISON:")
print(f"{'Metric':<15} {'Baseline':<10} {'XGBoost':<10} {'Improvement':<10}")
print(f"{'-'*45}")
print(f"{'MAP':<15} {baseline_poi_map:.4f}{'':<4} {xgb_poi_map:.4f}{'':<4} {((xgb_poi_map-baseline_poi_map)/baseline_poi_map*100):.1f}%")
print(f"{'MRR':<15} {baseline_poi_mrr:.4f}{'':<4} {xgb_poi_mrr:.4f}{'':<4} {((xgb_poi_mrr-baseline_poi_mrr)/baseline_poi_mrr*100):.1f}%")

# Get feature importance
print("\n" + "="*50)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Hotel feature importance
hotel_feature_importance = pd.DataFrame({
    'feature': hotel_features,
    'importance': hotel_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop Hotel Ranking Features:")
print(hotel_feature_importance.head(10))

# POI feature importance
poi_feature_importance = pd.DataFrame({
    'feature': poi_features,
    'importance': poi_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop POI Ranking Features:")
print(poi_feature_importance)

# Show top ranked hotels and POIs
print("\n" + "="*50)
print("TOP 10 RANKED HOTELS (XGBoost)")
print("="*50)
top_hotels = xgb_hotel_ranking[['Hotel Name', 'StarRating', 'PriceINR', 'xgb_score', 'relevance_score']].head(10)
print(top_hotels.to_string(index=False))

print("\n" + "="*50)
print("TOP 10 RANKED POIs (XGBoost)")
print("="*50)
top_pois = xgb_poi_ranking[['Place Name', 'Category', 'Distance_numeric', 'Entry_fee_numeric', 'xgb_score', 'poi_relevance']].head(10)
print(top_pois.to_string(index=False))


# Saving output as CSV file
xgb_hotel_ranking['final_rank'] = range(1, len(xgb_hotel_ranking) + 1)
xgb_poi_ranking['final_rank'] = range(1, len(xgb_poi_ranking) + 1)

hotel_output = xgb_hotel_ranking[[
    'final_rank',
    'Hotel Name',
    'StarRating',
    'PriceINR',
    'review_count',
    'xgb_score',
    'relevance_score'
]]

poi_output = xgb_poi_ranking[[
    'final_rank',
    'Place Name',
    'Category',
    'Distance_numeric',
    'Entry_fee_numeric',
    'xgb_score',
    'poi_relevance'
]]

hotel_output.to_csv('hotel_ranking_output.csv', index=False)
poi_output.to_csv('poi_ranking_output.csv', index=False)

print("\n✓ Hotel ranking saved to: hotel_ranking_output.csv")
print("✓ POI ranking saved to: poi_ranking_output.csv")



# Create comprehensive report
report = f"""
RANKING MODEL IMPROVEMENT REPORT
================================

SUMMARY:
- Hotel Ranking Model: Improved MAP by {((xgb_hotel_map-baseline_hotel_map)/baseline_hotel_map*100):.1f}%
- Hotel Ranking Model: Improved MRR by {((xgb_hotel_mrr-baseline_hotel_mrr)/baseline_hotel_mrr*100):.1f}%
- POI Ranking Model: Improved MAP by {((xgb_poi_map-baseline_poi_map)/baseline_poi_map*100):.1f}%
- POI Ranking Model: Improved MRR by {((xgb_poi_mrr-baseline_poi_mrr)/baseline_poi_mrr*100):.1f}%

MODEL FILES SAVED:
1. hotel_ranking_model.pkl - Hotel ranking model with {len(hotel_features)} features
2. poi_ranking_model.pkl - POI ranking model with {len(poi_features)} features

TOP HOTEL FEATURES (by importance):
{hotel_feature_importance.head(5).to_string(index=False)}

TOP POI FEATURES (by importance):
{poi_feature_importance.to_string(index=False)}

RECOMMENDATIONS:
1. For hotel ranking, focus on: {', '.join(hotel_feature_importance['feature'].head(3).tolist())}
2. For POI ranking, the most important feature is: {poi_feature_importance.iloc[0]['feature']}
3. Consider collecting more user interaction data for better relevance scores
4. Implement A/B testing to validate ranking improvements in production

TOP RANKED HOTELS:
{top_hotels.to_string(index=False)}

TOP RANKED POIs:
{top_pois.to_string(index=False)}
"""

print(report)

# Save report to file
with open('ranking_improvement_report.txt', 'w') as f:
    f.write(report)

print("\n✓ Report saved as: ranking_improvement_report.txt")
print("\n=== RANKING MODELS TRAINING COMPLETE ===")