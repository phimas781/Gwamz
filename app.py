import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Streamlit Config
st.set_page_config(page_title="Spotify Future Hits Predictor Pro", layout="wide", page_icon="🎧")

# Custom Theme Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        color: white; background: linear-gradient(90deg, #1DB954, #1ED760); border-radius: 8px;
    }
    .stSlider>div>div>div { background: linear-gradient(90deg, #1DB954, #1ED760); }
    .reportview-container .main footer { visibility: hidden; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🎧 Spotify Future Hits Predictor Pro Dashboard")
st.markdown("Upload Spotify CSV to predict track future popularity, potential hits, virality, and growth!")

# Sidebar File Upload
st.sidebar.header("Upload Spotify CSV")
uploaded_file = st.sidebar.file_uploader("Choose file", type="csv")

# Model Loading
reg_model = joblib.load('regression_model.pkl')
clf_model = joblib.load('classification_model.pkl')
le_album_type = joblib.load('le_album_type.pkl')
le_popularity_class = joblib.load('le_popularity_class.pkl')

# Features Required by Model
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

# Run if File Provided
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df_proc = preprocess(df.copy())
    X = df_proc[feature_cols]

    # Predictions
    df['Predicted_Popularity'] = reg_model.predict(X)
    df['Predicted_Class'] = le_popularity_class.inverse_transform(clf_model.predict(X))

    # Derived Metrics
    df['Album_Age_Years'] = 2025 - df['release_year']
    df['Followers_Popularity_Ratio'] = df['artist_followers'] / (df['artist_popularity'] + 1)
    df['Market_Reach_Score'] = df['available_markets_count'] / 180
    df['Disc_Track_Position_Score'] = df['track_number'] / (df['total_tracks_in_album'] + 1)
    df['Freshness_Score'] = 1 / (1 + df['Album_Age_Years'])
    df['Expected_Discoverability'] = df['Followers_Popularity_Ratio'] * df['Market_Reach_Score'] * df['Freshness_Score']

    # Predict Future Metrics
    df['Projected_Popularity_2026'] = df['Predicted_Popularity'] * (1 + 0.05 * df['Freshness_Score'])
    df['Virality_Score'] = df['Market_Reach_Score'] * df['Followers_Popularity_Ratio'] * 100
    df['Longevity_Score'] = (1 / (1 + df['Album_Age_Years'])) * df['Predicted_Popularity']
    df['Growth_Trend'] = df['Projected_Popularity_2026'] - df['Predicted_Popularity']

    # Data Preview
    st.subheader("🔍 Data with All Predictions & Metrics")
    st.dataframe(df)

    # Year Filter
    min_year, max_year = int(df['release_year'].min()), 2028
    release_range = st.sidebar.slider("Filter Release Year", min_year, max_year, (min_year, max_year))
    df_filtered = df[(df['release_year'] >= release_range[0]) & (df['release_year'] <= release_range[1])]

    # 🚀 Top 10 Future Hits
    st.subheader("🚀 Top 10 Future Hits (By Projected Popularity 2026)")
    top10 = df_filtered.sort_values(by='Projected_Popularity_2026', ascending=False).head(10)

    if not top10.empty:
        st.dataframe(top10[['track_name', 'Predicted_Popularity', 'Projected_Popularity_2026',
                            'Predicted_Class', 'Virality_Score', 'Longevity_Score', 'Growth_Trend']])
        
        # Key Insights
        top_proj = top10.iloc[0]
        top_viral = top10.sort_values(by='Virality_Score', ascending=False).iloc[0]
        top_long = top10.sort_values(by='Longevity_Score', ascending=False).iloc[0]

        st.markdown(f"""
        **🔑 Key Insights:**
        - 🎯 **Top Projected Hit:** {top_proj['track_name']} (Popularity 2026: {top_proj['Projected_Popularity_2026']:.2f})
        - 🔥 **Most Viral:** {top_viral['track_name']} (Virality Score: {top_viral['Virality_Score']:.2f})
        - 🕰️ **Longest Lifespan:** {top_long['track_name']} (Longevity Score: {top_long['Longevity_Score']:.2f})
        """)
    else:
        st.warning("No tracks in selected year range.")

    # 📊 Charts
    st.subheader("📊 Predicted Popularity Distribution")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df_filtered['Predicted_Popularity'], bins=15, kde=True, color='purple', ax=ax)
    ax.set_title('Predicted Popularity Distribution')
    st.pyplot(fig)

    st.subheader("🔮 Projected Popularity 2026")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df_filtered['Projected_Popularity_2026'], bins=15, kde=True, color='orange', ax=ax)
    ax.set_title('Projected Popularity for 2026')
    st.pyplot(fig)

    st.subheader("📈 Growth Trend Indicator")
    fig, ax = plt.subplots(figsize=(10,4))
    sns.histplot(df_filtered['Growth_Trend'], bins=15, kde=True, color='green', ax=ax)
    ax.set_title('Growth Trend of Tracks')
    st.pyplot(fig)

    st.subheader("🎯 Market Reach vs Discoverability")
    fig, ax = plt.subplots(figsize=(10,5))
    scatter = ax.scatter(df_filtered['Market_Reach_Score'], df_filtered['Expected_Discoverability'],
                         c=df_filtered['Predicted_Popularity'], cmap='viridis', s=60, alpha=0.7)
    ax.set_xlabel('Market Reach Score')
    ax.set_ylabel('Expected Discoverability')
    ax.set_title('Market Reach vs Discoverability')
    plt.colorbar(scatter, label='Predicted Popularity')
    st.pyplot(fig)

else:
    st.info("⬆️ Upload a Spotify CSV file to get predictions.")

# Footer
st.markdown("---")
st.markdown("Madness Mixtape 2025")
