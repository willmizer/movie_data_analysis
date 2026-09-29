import os
import gzip
import pickle
import joblib
import numpy as np
import streamlit as st

# paths
BASE_DIR             = os.path.dirname(__file__)
MODELS_DIR           = os.path.join(BASE_DIR, "models")
KNN_PATH             = os.path.join(MODELS_DIR, "knn_model.joblib")
FEATURE_MATRIX_PATH  = os.path.join(MODELS_DIR, "feature_matrix_reduced.npz")
DF_PATH              = os.path.join(MODELS_DIR, "movies_df.pkl.gz")  # gzipped DataFrame

# page config
st.set_page_config(
    page_title="MovieMatch AI",
    page_icon="🎬",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# verify model artifacts exist
for path, name in [
    (KNN_PATH, "KNN model"),
    (FEATURE_MATRIX_PATH, "feature matrix"),
    (DF_PATH, "metadata DataFrame")
]:
    if not os.path.exists(path):
        st.error(f"Error: {name} not found at {path}. Please run the training script.")
        st.stop()

# load and cache model
@st.cache_resource
def load_artifacts():
    # load KNN model
    knn = joblib.load(KNN_PATH)
    # load reduced matrix from .npz archive
    npzfile = np.load(FEATURE_MATRIX_PATH)
    X_red = npzfile['X_reduced']
    # load gzipped DataFrame
    with gzip.open(DF_PATH, 'rb') as f:
        df = pickle.load(f)
    df['genres_list'] = df['genres'].str.split(',')
    df['year'] = df['release_date'].str[:4]
    return knn, X_red, df

knn, X_red, df = load_artifacts()

# helper functions
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

# styling (mobile-first)
st.markdown("""
<style>
#MainMenu, footer, [data-testid="stSidebar"], [data-testid="collapsedControl"] {display: none;}
.block-container {max-width: 760px; padding: 2rem 1rem 3rem;}
.hero h1 {font-size: 2.2rem; margin: 0; padding: 0; line-height: 1.15;}
.hero p {opacity: .75; margin: .35rem 0 1.25rem;}
.rec-title {font-size: 1.1rem; font-weight: 700; margin: 0 0 .15rem;}
.rec-year {opacity: .6; font-weight: 400;}
.chips {display: flex; flex-wrap: wrap; gap: .35rem; margin: .4rem 0;}
.chip {background: rgba(128,128,128,.18); border-radius: 999px; padding: .1rem .6rem; font-size: .78rem;}
.stats {font-size: .9rem; opacity: .85; margin: .3rem 0;}
div[data-testid="stForm"] {border: none; padding: 0;}
.stButton button, .stFormSubmitButton button {width: 100%; min-height: 2.9rem; border-radius: 10px;}
div[data-testid="stVerticalBlockBorderWrapper"] {border-radius: 14px;}
@media (max-width: 640px) {
  .block-container {padding: 1rem .75rem 2rem;}
  .hero h1 {font-size: 1.7rem;}
  /* keep poster + details side by side on phones instead of stacking */
  div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .75rem;}
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {flex: 0 0 32% !important; min-width: 0 !important;}
  div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {flex: 1 1 0 !important; min-width: 0 !important;}
}
</style>
""", unsafe_allow_html=True)

# header
st.markdown(
    '<div class="hero"><h1>🎬 MovieMatch AI</h1>'
    '<p>Tell us a movie you love. We\'ll find what to watch next.</p></div>',
    unsafe_allow_html=True,
)

# search (form so Enter / mobile "Go" submits)
with st.form("search_form", border=False):
    query = st.text_input(
        "Movie title", placeholder="e.g. Inception", label_visibility="collapsed"
    )
    submitted = st.form_submit_button("Search", type="primary")

if submitted:
    st.session_state.query = query.strip()
query = st.session_state.get("query", "")


def render_rec(row):
    with st.container(border=True):
        c1, c2 = st.columns([1, 3], vertical_alignment="top")
        poster = row.get('poster_url')
        if isinstance(poster, str) and poster.startswith("http"):
            c1.image(poster, use_container_width=True)
        with c2:
            st.markdown(
                f"<div class='rec-title'>{row['title']} "
                f"<span class='rec-year'>({row['year']})</span></div>"
                f"<div class='stats'>⭐ {row['vote_average']:.1f} &nbsp;·&nbsp; "
                f"💰 {fmt_profit(row.get('profit_in_millions'))} &nbsp;·&nbsp; "
                f"⏱ {fmt_runtime(row.get('runtime', ''))}</div>"
                "<div class='chips'>"
                + "".join(f"<span class='chip'>{g.strip()}</span>" for g in row['genres_list'])
                + "</div>",
                unsafe_allow_html=True,
            )
        with st.expander("Overview"):
            st.write(row.get('overview') or "No overview available.")


if not query:
    st.caption("Search by title to get started.")
else:
    matches = df[df['title'].str.contains(query, case=False, na=False, regex=False)]
    suggestions = (
        matches[['title', 'year']]
               .drop_duplicates()
               .assign(label=lambda d: d['title'] + ' (' + d['year'] + ')')
               .sort_values('label')['label']
    )
    if suggestions.empty:
        st.warning(f"No matches found for \"{query}\".")
    else:
        choice = st.selectbox("Select a movie", suggestions)
        title, year = choice.rsplit(' (', 1)
        year = year.rstrip(')')
        idx = df[(df['title'] == title) & (df['year'] == year)].index[0]
        _, indices = knn.kneighbors(X_red[idx].reshape(1, -1), n_neighbors=6)
        recs = df.iloc[indices[0][1:]]

        st.subheader(f"Because you liked {title} ({year})")
        for _, row in recs.iterrows():
            render_rec(row)
