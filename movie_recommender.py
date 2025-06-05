import streamlit as st
from streamlit_js_eval import streamlit_js_eval
import pandas as pd
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
import datetime

# Page Configuration
st.set_page_config(page_title="MovieMatch AI")


# Load Models and Data
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("models/tfidf_genre.pkl", "rb") as f:
    tfidf_genre = pickle.load(f)
with open("models/tfidf_cast.pkl", "rb") as f:
    tfidf_cast = pickle.load(f)
with open("models/tfidf_director.pkl", "rb") as f:
    tfidf_director = pickle.load(f)
with open("models/tfidf_writer.pkl", "rb") as f:
    tfidf_writer = pickle.load(f)
with open("models/tfidf_producer.pkl", "rb") as f:
    tfidf_producer = pickle.load(f)
with open("models/tfidf_prod_comp.pkl", "rb") as f:
    tfidf_prod_comp = pickle.load(f)
with open("models/overview_features.npy", "rb") as f:
    overview_features = np.load(f)
with open("models/knn_model.pkl", "rb") as f:
    knn = pickle.load(f)
with open("models/reduced_vector.npy", "rb") as f:
    reduced_vector = np.load(f)

@st.cache_data
def load_data():
    with open("models/movies_df.pkl", "rb") as f:
        return pickle.load(f)

df = load_data()

# Helper Functions
def format_profit(profit):
    try:
        profit = float(profit)
        return f"{profit / 1000:.2f}B" if profit >= 1000 else f"{profit:.2f}M"
    except:
        return "N/A"

def format_runtime(minutes):
    try:
        minutes = int(minutes)
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours}h{mins:02d}m"
    except:
        return "N/A"

def clean_genres(genre_str):
    try:
        genres = eval(genre_str) if isinstance(genre_str, str) else genre_str
        return ", ".join(genres) if isinstance(genres, list) else "N/A"
    except:
        return "N/A"

def get_unique_genres(df):
    genre_set = set()
    for genres in df['genres']:
        if isinstance(genres, str):
            try:
                genres = eval(genres)
            except:
                genres = []
        if isinstance(genres, list):
            genre_set.update(genres)
    return sorted(genre_set)

unique_genres = get_unique_genres(df)

# ---------------- UI ----------------
st.title("🎬 MovieMatch AI — Beta")
st.caption("Built for discovery.")

col1, col2 = st.columns([3, 1])
with col1:
    typed_title = st.text_input("Search for a movie you love:")
with col2:
    st.session_state.exclude_genres = st.multiselect("Exclude genres:", unique_genres, default=st.session_state.get("exclude_genres", []))

filtered_titles = df[df['title'].str.lower().str.contains(typed_title.lower(), na=False)]
suggestions = (filtered_titles['title'] + " (" + filtered_titles['year'].astype(str) + ")").drop_duplicates().sort_values()

if suggestions.any() and typed_title.strip():
    selected_suggestion = st.selectbox("Select your movie:", suggestions.tolist(), key="suggestbox")
    if st.button("Get Recommendations"):
        title_part, year_part = selected_suggestion.rsplit(" (", 1)
        year_part = int(year_part.strip(")"))
        matched_row = df[(df['title'] == title_part) & (df['year'] == year_part)]
        if not matched_row.empty:
            st.session_state.selected_idx = df.index.get_loc(matched_row.index[0])
            st.session_state.confirmed = True
            st.rerun()

# ---------------- Recommendations ----------------
if st.session_state.get("confirmed") and st.session_state.get("selected_idx") is not None:
    distances, indices = knn.kneighbors([reduced_vector[st.session_state.selected_idx]], n_neighbors=6)
    results = df.iloc[indices[0][1:]].copy()

    if st.session_state.exclude_genres:
        results = results[~results['genres'].apply(lambda g: any(ex in g for ex in st.session_state.exclude_genres))]

    st.markdown(f"### Because you liked **{df.loc[st.session_state.selected_idx, 'title']} ({df.loc[st.session_state.selected_idx, 'year']})**")
    for _, row in results.head(5).iterrows():
        with st.container():
            st.image(row['poster_url'], width=150)
            st.markdown(f"**{row['title']}** ({row['year']})")
            st.caption(f"⭐ {row['average_rating']:.2f} | 💰 {format_profit(row['profit_in_millions'])} | 🎭 {clean_genres(row['genres'])} | ⏱ {format_runtime(row.get('runtime', 'N/A'))}")
            with st.expander("Overview"):
                st.markdown(row.get('overview', 'No description available.'))
            st.divider()

    st.session_state.confirmed = False
    st.session_state.selected_idx = None

# ---------------- Reset Button ----------------
if st.button("🔁 Start Over"):
    streamlit_js_eval(js_expressions="parent.window.location.reload()")
