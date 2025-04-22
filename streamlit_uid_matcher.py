
# Streamlit App: UID Matching and Survey Template Parsing

import streamlit as st
import pandas as pd
import re
import uuid
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

# --- Helper Functions ---
synonym_map = {
    "please select": "what is",
    "sector you are from": "your sector",
    "identity type": "id type",
    "what type of": "type of",
    "are you": "do you",
}

def apply_synonyms(text):
    for phrase, replacement in synonym_map.items():
        text = text.replace(phrase, replacement)
    return text

def enhanced_normalize(text):
    text = str(text).lower()
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'[^a-z0-9 ]', '', text)
    text = apply_synonyms(text)
    words = text.split()
    return ' '.join([w for w in words if w not in ENGLISH_STOP_WORDS])

def tfidf_match(df_mapped, df_unmapped):
    df_mapped["norm_text"] = df_mapped["heading_0"].apply(enhanced_normalize)
    df_unmapped["norm_text"] = df_unmapped["heading_0"].apply(enhanced_normalize)
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    vectorizer.fit(df_unmapped["norm_text"].tolist() + df_mapped["norm_text"].tolist())

    sim_matrix = cosine_similarity(
        vectorizer.transform(df_unmapped["norm_text"]),
        vectorizer.transform(df_mapped["norm_text"])
    )
    
    matches, scores = [], []
    for i, row in enumerate(sim_matrix):
        best_idx = row.argmax()
        score = row[best_idx]
        uid = df_mapped.iloc[best_idx]["uid"] if score >= 0.5 else None
        matches.append(uid)
        scores.append(round(score, 4))

    df_unmapped["Matched_UID"] = matches
    df_unmapped["Similarity_Score"] = scores
    return df_unmapped

# --- Streamlit UI ---
st.title("🧠 UID Matcher for Survey Templates")

st.sidebar.markdown("## 📁 Upload Files")
uid_file = st.sidebar.file_uploader("Upload UID Reference File (CSV)", type="csv")
survey_file = st.sidebar.file_uploader("Upload Survey Template File (CSV)", type="csv")

if uid_file and survey_file:
    df_mapped = pd.read_csv(uid_file)
    df_unmapped = pd.read_csv(survey_file)

    if "heading_0" not in df_mapped or "uid" not in df_mapped.columns:
        st.error("⚠️ UID Reference file must contain columns: 'heading_0' and 'uid'")
    elif "heading_0" not in df_unmapped:
        st.error("⚠️ Survey template file must contain column: 'heading_0'")
    else:
        st.success("✅ Files loaded. Ready to match questions.")

        if st.button("🔍 Match Questions"):
            matched_df = tfidf_match(df_mapped, df_unmapped)
            st.subheader("🔗 Matched Results")
            st.dataframe(matched_df[["heading_0", "Matched_UID", "Similarity_Score"]])

            csv = matched_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Matched Results as CSV", csv, "matched_results.csv", "text/csv")
else:
    st.info("👈 Upload both files to begin UID matching.")
