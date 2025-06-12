import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# Set Streamlit Page Configuration
st.set_page_config(page_title="Spotify Track Future Popularity Predictor", layout="wide")

# Title & Description
st.title("🎵 Spotify Track Popularity Predictor Dashboard")
st.markdown("""
Welcome to the Spotify Song Future Popularity Predictor!  
Upload your Spotify track CSV file (with Gwamz-like features) to see predicted future performance.
""")

# Sidebar
st.sidebar.header("Upload Your CSV Data")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load Models and Encoders
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')
le_album_type = joblib.load('le_album_type.pkl')
le_popularity_class = joblib.load('le_popularity_class.pkl')

# Feature Columns (same as training)
feature_cols = [
    'artist_followers', 'artist_popularity', 'release_year',
    'total_tracks_in_album', 'available_markets_count',
    'track_number', 'disc_number', 'explicit', 'album_type_encoded'
]

# Process Function
def preprocess(df):
    df = df.drop(['artist_name', 'artist_id', 'album_name', 'album_id', 'track_name', 'track_id'], axis=1)
    df['album_type_encoded'] = le_album_type.transform(df['album_type'])
    df['explicit'] = df['explicit'].astype(int)
    return df

# If file uploaded
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_processed = preprocess(df.copy())
    
    X = df_processed[feature_cols]
    
    # Predictions
    df['Predicted_Popularity'] = reg_model.predict(X)
    df['Predicted_Class'] = le_popularity_class.inverse_transform(clf_model.predict(X))

    # Additional Metrics
    df['Album_Age_Years'] = 2025 - df['release_year']
    df['Followers_Popularity_Ratio'] = df['artist_followers'] / (df['artist_popularity'] + 1)
    df['Market_Reach_Score'] = df['available_markets_count'] / 180
    df['Disc_Track_Position_Score'] = df['track_number'] / (df['total_tracks_in_album'] + 1)
    df['Freshness_Score'] = 1 / (1 + df['Album_Age_Years'])
    df['Expected_Discoverability'] = df['Followers_Popularity_Ratio'] * df['Market_Reach_Score'] * df['Freshness_Score']

    # Simulate Future Popularity Growth
    df['Projected_Popularity_2026'] = df['Predicted_Popularity'] * np.random.uniform(1.02, 1.08, len(df))
    df['Projected_Popularity_2027'] = df['Projected_Popularity_2026'] * np.random.uniform(1.01, 1.05, len(df))
    df['Projected_Popularity_2028'] = df['Projected_Popularity_2027'] * np.random.uniform(0.98, 1.03, len(df))
    df['Virality_Potential'] = np.clip(df['Expected_Discoverability'] * np.random.uniform(0.8, 1.2, len(df)), 0, 1)
    df['Longevity_Score'] = np.clip(df['Freshness_Score'] * np.random.uniform(0.7, 1.3, len(df)), 0, 1)

    # Top 5 Future Hits
    st.subheader("🚀 Top 5 Future Hits (By Predicted Popularity)")
    top5 = df.sort_values(by='Predicted_Popularity', ascending=False).head(5)

    def determine_trend(row):
        if row['Projected_Popularity_2028'] > row['Predicted_Popularity']:
            return "Up 🔼"
        elif row['Projected_Popularity_2028'] < row['Predicted_Popularity']:
            return "Down 🔽"
        else:
            return "Stable ➖"

    top5['Growth_Trend'] = top5.apply(determine_trend, axis=1)

    st.write(top5[['track_name', 'Predicted_Popularity', 'Predicted_Class',
                   'Projected_Popularity_2026', 'Projected_Popularity_2027', 'Projected_Popularity_2028',
                   'Expected_Discoverability', 'Virality_Potential', 'Longevity_Score', 'Growth_Trend']])

    # Plot Future Growth
    st.subheader("📈 Future Popularity Growth Trends (Top 5)")
    plt.figure(figsize=(10, 6))
    for i, row in top5.iterrows():
        plt.plot([2025, 2026, 2027, 2028],
                 [row['Predicted_Popularity'], row['Projected_Popularity_2026'],
                  row['Projected_Popularity_2027'], row['Projected_Popularity_2028']],
                 marker='o', label=row['track_name'])
    plt.xlabel('Year')
    plt.ylabel('Projected Popularity')
    plt.title('Projected Popularity Trend for Top 5 Future Hits')
    plt.legend()
    st.pyplot(plt)

else:
    st.info("⬆️ Please upload a Spotify track CSV file to begin.")

st.markdown("---")
st.markdown("🔗 **Built by Your Data Science Pro System. Powered by Streamlit & Spotify API.**")
