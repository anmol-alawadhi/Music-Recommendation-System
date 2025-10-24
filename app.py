
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Music Recommendation System", page_icon="🎧", layout="wide")

st.title("🎧 Music Recommendation System")
st.write("Content-based recommendations using genres and mood tags.")

@st.cache_data
def load_data():
    df = pd.read_csv("songs.csv")
    df["tags"] = (df["genres"].fillna("") + " " + df["mood"].fillna("")).str.strip()
    return df

df = load_data()

    # Build vector space
@st.cache_resource
def build_model(corpus):
    vec = TfidfVectorizer(ngram_range=(1,2), min_df=1)
    X = vec.fit_transform(corpus)
    return vec, X

vec, X = build_model(df["tags"].tolist())

    # Sidebar
st.sidebar.header("Search / Filter")
choice = st.sidebar.selectbox("Pick a seed track", df["title"] + " — " + df["artist"])
top_k = st.sidebar.slider("Number of recommendations", 3, 10, 5)

    # Optional free-text mood/genre query
user_query = st.text_input("Describe what you want to listen to (e.g., 'chill lofi for coding')")
if st.button("Recommend"):
    if user_query.strip():
        q = vec.transform([user_query.strip()])
        sims = cosine_similarity(q, X).ravel()
    else:
            # use chosen seed track
        seed_idx = (df["title"] + " — " + df["artist"]).tolist().index(choice)
        sims = cosine_similarity(X[seed_idx], X).ravel()

        # Exclude the exact seed when using seed track
    if not user_query.strip():
        seed_idx = (df["title"] + " — " + df["artist"]).tolist().index(choice)
        sims[seed_idx] = -1

    top_idx = np.argsort(-sims)[:top_k]
    st.subheader("Top Recommendations")
    for rank, i in enumerate(top_idx, start=1):
        st.markdown(f"**#{rank}. {df.loc[i, 'title']}** — {df.loc[i, 'artist']}  
"
                        f"Genres: `{df.loc[i, 'genres']}` | Mood: `{df.loc[i, 'mood']}`")

with st.expander("Dataset Preview"):
    st.dataframe(df)

st.markdown("""---
    ### Deployment
    - **Streamlit Cloud**: push this repo to GitHub, then deploy at https://share.streamlit.io  
    - **Local run**: `pip install -r requirements.txt && streamlit run app.py`
    """)
