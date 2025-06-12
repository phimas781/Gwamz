import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Set Streamlit Page Configuration
st.set_page_config(page_title="Spotify Track Future Popularity Predictor", layout="wide", page_icon="🎧")

# Apply Streamlit Custom Theme
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stButton>button {
        color: white;
        background: linear-gradient(90deg, #1DB954, #1ed760);
        border-radius: 8px;
    }
    .stSlider>div>div>div {
        background: linear-gradient(90deg, #1DB954, #1ed760);
    }
    </style>
""", unsafe_allow_html=True)

# Title & Description
st.title("🎧 Spotify Track Popularity Predictor Dashboard")
st.markdown("""
Welcome to the **Spotify Song Future Popularity Predictor Pro Version** 🎶 

Upload your Spotify track CSV file to see predictions, future hit potential, virality, longevity, and growth trends.
""")

# Sidebar Upload
st.sidebar.header("Upload Your Spotify CSV")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Load Models
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')
le_album_type = joblib.load('le_album_type.pkl')
le_popularity_class = joblib.load('le_popularity_class.pkl')

feature_cols = [
    'artist_followers', 'artist_popularity', 'release_year',
    'total_tracks_in_album', 'available_markets_count',
    'track_number', 'disc_number', 'explicit', 'album_type_encoded'
]

def preprocess(df):
    df = df.drop(['artist_name', 'artist_id', 'album_name', 'album_id', 'track_name', 'track_id'], axis=1)
    df['album_type_encoded'] = le_album_type.transform(df['album_type'])
    df['explicit'] = df['explicit'].astype(int)
    return df

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_processed = preprocess(df.copy())
    X = df_processed[feature_cols]

    # Predictions
    df['Predicted_Popularity'] = reg_model.predict(X)
    df['Predicted_Class'] = le_popularity_class.inverse_transform(clf_model.predict(X))

    # Extra Metrics
    df['Album_Age_Years'] = 2025 - df['release_year']
    df['Followers_Popularity_Ratio'] = df['artist_followers'] / (df['artist_popularity'] + 1)
    df['Market_Reach_Score'] = df['available_markets_count'] / 180
    df['Disc_Track_Position_Score'] = df['track_number'] / (df['total_tracks_in_album'] + 1)
    df['Freshness_Score'] = 1 / (1 + df['Album_Age_Years'])
    df['Expected_Discoverability'] = df['Followers_Popularity_Ratio'] * df['Market_Reach_Score'] * df['Freshness_Score']

    # Future Metrics
    df['Projected_Popularity_2026'] = df['Predicted_Popularity'] * (1 + 0.05 * df['Freshness_Score'])
    df['Virality_Score'] = df['Market_Reach_Score'] * df['Followers_Popularity_Ratio'] * 100
    df['Longevity_Score'] = (1 / (1 + df['Album_Age_Years'])) * df['Predicted_Popularity']
    df['Growth_Trend'] = df['Projected_Popularity_2026'] - df['Predicted_Popularity']

    # Data Preview
    st.subheader("🔍 Data Preview with All Predictions & Metrics")
    st.dataframe(df)

    # Year Filter
    min_year, max_year = int(df['release_year'].min()), 2028
    release_year_range = st.sidebar.slider("Filter by Release Year", min_year, max_year, (min_year, max_year))
    df_filtered = df[(df['release_year'] >= release_year_range[0]) & (df['release_year'] <= release_year_range[1])]

    # 🚀 Future Hits Analysis
    st.subheader("🚀 Top 5 Future Hits Analysis")
    top5 = df_filtered.sort_values(by='Projected_Popularity_2026', ascending=False).head(5)
    st.write(top5[['track_name', 'Predicted_Popularity', 'Projected_Popularity_2026', 'Predicted_Class', 'Virality_Score', 'Longevity_Score', 'Growth_Trend']])

    # Charts
    st.subheader("📊 Popularity Distribution")
    plt.figure(figsize=(10, 4))
    sns.histplot(df_filtered['Predicted_Popularity'], bins=15, kde=True, color='purple')
    plt.title('Predicted Popularity Distribution')
    st.pyplot(plt)

    st.subheader("🔮 Projected Popularity 2026")
    plt.figure(figsize=(10, 4))
    sns.histplot(df_filtered['Projected_Popularity_2026'], bins=15, kde=True, color='orange')
    plt.title('Projected Popularity for 2026')
    st.pyplot(plt)

    st.subheader("📈 Growth Trend Indicator")
    plt.figure(figsize=(10, 4))
    sns.histplot(df_filtered['Growth_Trend'], bins=15, kde=True, color='green')
    plt.title('Growth Trend of Tracks')
    st.pyplot(plt)

else:
    st.info("⬆️ Please upload a Spotify CSV to begin.")

# Footer
st.markdown("---")
st.markdown("Made with ❤️ by Your Data Science Pro System | Powered by Streamlit & Spotify API.")
