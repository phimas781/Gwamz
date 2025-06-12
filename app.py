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
    df['Market_Reach_Score'] = df['available_markets_count'] / 180  # Assuming 180 total markets
    df['Disc_Track_Position_Score'] = df['track_number'] / (df['total_tracks_in_album'] + 1)
    df['Freshness_Score'] = 1 / (1 + df['Album_Age_Years'])
    df['Expected_Discoverability'] = df['Followers_Popularity_Ratio'] * df['Market_Reach_Score'] * df['Freshness_Score']

    # Show Data
    st.subheader("🔍 Data Preview with Predictions & Metrics")
    st.dataframe(df)

    # Sidebar filter by Release Year
    min_year, max_year = int(df['release_year'].min()), int(df['release_year'].max())
    release_year_range = st.sidebar.slider("Filter by Release Year", min_year, max_year, (min_year, max_year))
    df_filtered = df[(df['release_year'] >= release_year_range[0]) & (df['release_year'] <= release_year_range[1])]

    # Top 5 Future Hits
    st.subheader("🚀 Top 5 Future Hits (By Predicted Popularity)")
    top5 = df_filtered.sort_values(by='Predicted_Popularity', ascending=False).head(5)
    st.write(top5[['track_name', 'Predicted_Popularity', 'Predicted_Class']])

    # Popularity Distribution Chart
    st.subheader("📊 Predicted Popularity Distribution")
    plt.figure(figsize=(10, 5))
    sns.histplot(df_filtered['Predicted_Popularity'], bins=20, kde=True, color='purple')
    plt.xlabel('Predicted Popularity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Track Popularity')
    st.pyplot(plt)

    # Expected Discoverability Chart
    st.subheader("🔮 Expected Discoverability Score Distribution")
    plt.figure(figsize=(10, 5))
    sns.histplot(df_filtered['Expected_Discoverability'], bins=20, kde=True, color='green')
    plt.xlabel('Expected Discoverability Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Expected Discoverability')
    st.pyplot(plt)

else:
    st.info("⬆️ Please upload a Spotify track CSV file to begin.")

# Footer
st.markdown("---")
st.markdown("🔗 **Built by Your Data Science Pro System. Powered by Streamlit & Spotify API.**")
