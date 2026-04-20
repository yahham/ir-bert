indexMapping = {
    "properties": {
        # BEIR document ID (e.g. "4983", "13728")
        "doc_id": {
            "type": "keyword"
        },
        # Title of the scientific abstract
        "title": {
            "type": "text",
            "analyzer": "english"
        },
        # Full abstract text
        "text": {
            "type": "text",
            "analyzer": "english"
        },
        # Combined title + text, used for display
        "full_text": {
            "type": "text",
            "analyzer": "english"
        },
        # 768-dim BERT embedding of full_text
        "embedding": {
            "type": "dense_vector",
            "dims": 768,
            "index": True,
            "similarity": "cosine"
        }
    }
}
