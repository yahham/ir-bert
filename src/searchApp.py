import os
import streamlit as st
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

load_dotenv()

INDEX_NAME = "beir_scifact"

# ── Connect ────────────────────────────────────────────────────────────────────

try:
    es = Elasticsearch(
        os.getenv("ES_HOST"),
        basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
        ca_certs=os.getenv("ES_CA_CERT"),
        request_timeout=60,
        max_retries=3,
        retry_on_timeout=True,
    )
except Exception as e:
    st.error(f"Elasticsearch connection error: {e}")
    st.stop()

# Cache the model so it is loaded only once across Streamlit reruns
@st.cache_resource
def load_model():
    return SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# ── Search function ────────────────────────────────────────────────────────────

def search(query_text: str, top_k: int = 10):
    model        = load_model()
    query_vector = model.encode(
        query_text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    resp = es.search(
        index=INDEX_NAME,
        knn={
            "field":          "embedding",
            "query_vector":   query_vector,
            "k":              top_k,
            "num_candidates": top_k * 5,
        },
        size=top_k,
        source=["doc_id", "title", "text"],
    )
    return resp["hits"]["hits"]

# ── Streamlit UI ───────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="BERT Search — SciFact", layout="wide")
    st.title("🔬 BERT Semantic Search — SciFact")
    st.caption(
        "Uses `sentence-transformers/all-mpnet-base-v2` embeddings "
        "stored in Elasticsearch with cosine KNN search."
    )

    query_text = st.text_input("Enter your scientific query:", placeholder="e.g. smoking causes lung cancer")
    top_k      = st.slider("Number of results", min_value=1, max_value=20, value=10)

    if st.button("Search") and query_text.strip():
        with st.spinner("Encoding query and searching..."):
            results = search(query_text.strip(), top_k=top_k)

        if not results:
            st.warning("No results found.")
            return

        st.subheader(f"Top {len(results)} results")
        for i, hit in enumerate(results, start=1):
            src   = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            with st.container():
                st.markdown(f"**{i}. {src.get('title', 'No title')}**")
                st.caption(f"Doc ID: `{src.get('doc_id', hit['_id'])}` | Cosine similarity: `{score:.4f}`")
                st.write(src.get("text", ""))
                st.divider()

if __name__ == "__main__":
    main()
