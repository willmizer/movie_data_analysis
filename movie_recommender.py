import os
import gzip
import pickle
import joblib
import numpy as np
import streamlit as st

# ─── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR             = os.path.dirname(__file__)
MODELS_DIR           = os.path.join(BASE_DIR, "models")
KNN_PATH             = os.path.join(MODELS_DIR, "knn_model.joblib")
FEATURE_MATRIX_PATH  = os.path.join(MODELS_DIR, "feature_matrix_reduced.npz")
DF_PATH              = os.path.join(MODELS_DIR, "movies_df.pkl.gz")  # gzipped DataFrame

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MovieMatch AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Verify Artifacts ──────────────────────────────────────────────────────────
for path, name in [
    (KNN_PATH, "KNN model"),
    (FEATURE_MATRIX_PATH, "feature matrix"),
    (DF_PATH, "metadata DataFrame")
]:
    if not os.path.exists(path):
        st.error(f"Error: {name} not found at {path}. Please run the training script.")
        st.stop()

# ─── Load & Cache Artifacts ─────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    # Load KNN
    knn = joblib.load(KNN_PATH)
    # Load reduced matrix from .npz archive
    npzfile = np.load(FEATURE_MATRIX_PATH)
    X_red = npzfile['X_reduced']
    # Load gzipped DataFrame
    with gzip.open(DF_PATH, 'rb') as f:
        df = pickle.load(f)
    df['genres_list'] = df['genres'].str.split(',')
    df['year'] = df['release_date'].str[:4]
    return knn, X_red, df

knn, X_red, df = load_artifacts()

# ─── Helpers ───────────────────────────────────────────────────────────────────
def fmt_profit(val):
    try:
        v = float(val)
        return f"${v/1e3:.1f}B" if v >= 1e3 else f"${v:.1f}M"
    except:
        return "N/A"


def fmt_runtime(x):
    try:
        m = int(x)
        h, m = divmod(m, 60)
        return f"{h}h {m:02d}m"
    except:
        return "N/A"

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.header("Find Similar Movies")
query = st.sidebar.text_input("Movie title")
if st.sidebar.button("Search"):
    st.session_state.search = True
if st.sidebar.button("Clear Search"):
    st.sidebar.info("Search cleared.")
    st.session_state.search = False

# ─── Main ──────────────────────────────────────────────────────────────────────
st.title("MovieMatch AI - Beta")
st.markdown("Personalized movie recommendations powered by transformers & KNN.")

if st.session_state.get('search', False) and query:
    # Filter titles
    matches = df[df['title'].str.contains(query, case=False, na=False)]
    suggestions = (
        matches[['title','year']]
               .drop_duplicates()
               .assign(label=lambda d: d['title'] + ' (' + d['year'] + ')')
               .sort_values('label')['label']
    )
    if suggestions.empty:
        st.warning("No matches found.")
    else:
        choice = st.selectbox("Select a movie:", suggestions)
        if st.button("Get Recommendations"):
            title, year = choice.rsplit(' (', 1)
            year = year.rstrip(')')
            idx = df[(df['title'] == title) & (df['year'] == year)].index[0]
            # Query KNN (reshape to 2D)
            q_vec = X_red[idx].reshape(1, -1)
            distances, indices = knn.kneighbors(q_vec, n_neighbors=6)
            recs = df.iloc[indices[0][1:]].copy()

            st.subheader(f"Because you liked **{title} ({year})**")
            for _, row in recs.head(3).iterrows():
                c1, c2 = st.columns([1,5])
                c1.image(row['poster_url'], use_container_width=True)
                c2.markdown(f"**{row['title']}** ({row['year']})")
                c2.markdown(
                    f"⭐ {row['vote_average']:.1f}  |  💰 {fmt_profit(row.get('profit_in_millions'))}  |  ⏱ {fmt_runtime(row.get('runtime',''))}"
                )
                c2.text(f"Genres: {', '.join(row['genres_list'])}")
                with c2.expander("Overview"):
                    st.write(row.get('overview','No overview available.'))
                st.markdown("---")
else:
    st.info("Use the sidebar to search for a movie.")