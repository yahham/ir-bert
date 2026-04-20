import os
import json
import math
import logging
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

INDEX_NAME   = "beir_scifact"
DATASET_PATH = os.getenv("BEIR_DATASET_PATH", "../../datasets/scifact")
QUERIES_FILE = os.path.join(DATASET_PATH, "queries.jsonl")
QRELS_FILE   = os.path.join(DATASET_PATH, "qrels", "test.tsv")
TOP_K        = 100   # retrieve this many results per query

# ── Connect ────────────────────────────────────────────────────────────────────

es = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
    ca_certs=os.getenv("ES_CA_CERT"),
    request_timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)
if not es.ping():
    raise RuntimeError("Cannot connect to Elasticsearch.")
logger.info("Connected to Elasticsearch.")

# ── Load queries ───────────────────────────────────────────────────────────────

logger.info(f"Loading queries from {QUERIES_FILE} ...")
queries = {}
with open(QUERIES_FILE, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        q = json.loads(line)
        queries[q["_id"]] = q["text"]
logger.info(f"  {len(queries)} queries loaded.")

# ── Load qrels ─────────────────────────────────────────────────────────────────
# qrels[query_id][doc_id] = relevance_score  (only score > 0 are stored)

logger.info(f"Loading qrels from {QRELS_FILE} ...")
qrels = {}
with open(QRELS_FILE, "r", encoding="utf-8") as f:
    next(f)   # skip header line: "query-id\tcorpus-id\tscore"
    for line in f:
        parts = line.strip().split("\t")
        if len(parts) < 3:
            continue
        qid, did, score = parts[0], parts[1], int(parts[2])
        if score > 0:
            qrels.setdefault(qid, {})[did] = score
logger.info(f"  {len(qrels)} queries have relevance judgements.")

# ── Load BERT model ────────────────────────────────────────────────────────────

logger.info("Loading BERT model...")
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
logger.info("Model loaded.")

# ── Run queries and collect ranked results ─────────────────────────────────────
#
# Only evaluate queries that appear in qrels — same as the Rust evaluator.

queries_to_eval = {qid: text for qid, text in queries.items() if qid in qrels}
logger.info(f"Evaluating {len(queries_to_eval)} queries...")

ranked_results = {}   # query_id -> [doc_id, doc_id, ...]  ordered by score desc

for qid, qtext in tqdm(queries_to_eval.items(), desc="Querying"):
    query_vector = model.encode(
        qtext,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).tolist()

    resp = es.search(
        index=INDEX_NAME,
        knn={
            "field":        "embedding",
            "query_vector": query_vector,
            "k":            TOP_K,
            "num_candidates": TOP_K * 5,   # candidates > k for better recall
        },
        size=TOP_K,
        source=False,   # we only need the _id, not the full document
    )

    ranked_results[qid] = [hit["_id"] for hit in resp["hits"]["hits"]]

# ── Metric functions ───────────────────────────────────────────────────────────

def mrr_at_k(ranked_results, qrels, k):
    """
    Mean Reciprocal Rank @ k.
    For each query: find rank of first relevant doc in top-k, take 1/rank.
    Average across all queries.
    """
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        for rank, did in enumerate(doc_ids[:k], start=1):
            if did in rel:
                total += 1.0 / rank
                break
    return total / n if n > 0 else 0.0


def ndcg_at_k(ranked_results, qrels, k):
    """
    Normalized Discounted Cumulative Gain @ k.
    DCG  = Σ rel_i / log2(i+2)   for i in 0..k
    IDCG = DCG of the ideal ordering
    NDCG = DCG / IDCG
    """
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1

        dcg = sum(
            rel.get(did, 0) / math.log2(i + 2)
            for i, did in enumerate(doc_ids[:k])
        )

        ideal = sorted(rel.values(), reverse=True)
        idcg  = sum(
            r / math.log2(i + 2)
            for i, r in enumerate(ideal[:k])
        )

        if idcg > 0:
            total += dcg / idcg

    return total / n if n > 0 else 0.0


def recall_at_k(ranked_results, qrels, k):
    """
    Recall @ k.
    Fraction of ALL relevant documents that appear in the top-k results.
    """
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        hits = sum(1 for did in doc_ids[:k] if did in rel)
        total += hits / len(rel)
    return total / n if n > 0 else 0.0


def precision_at_k(ranked_results, qrels, k):
    """
    Precision @ k.
    Fraction of the top-k results that are relevant.
    """
    total, n = 0.0, 0
    for qid, doc_ids in ranked_results.items():
        rel = qrels.get(qid, {})
        if not rel:
            continue
        n += 1
        top_k = doc_ids[:k]
        if not top_k:
            continue
        hits = sum(1 for did in top_k if did in rel)
        total += hits / len(top_k)
    return total / n if n > 0 else 0.0


def f1_at_k(ranked_results, qrels, k):
    """
    F1 @ k.
    Harmonic mean of Precision@k and Recall@k.
    """
    p = precision_at_k(ranked_results, qrels, k)
    r = recall_at_k(ranked_results, qrels, k)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


# ── Print results table ────────────────────────────────────────────────────────

ks = [1, 5, 10, 20, 100]

print("\n" + "═" * 70)
print("  BERT (all-mpnet-base-v2) Evaluation on BEIR SciFact")
print(f"  Corpus: 5183 docs  |  Queries evaluated: {len(ranked_results)}")
print("═" * 70)
print(f"\n{'Metric':<15}" + "".join(f"{'@'+str(k):>10}" for k in ks))
print("-" * 65)

metrics = [
    ("MRR",       mrr_at_k),
    ("NDCG",      ndcg_at_k),
    ("Recall",    recall_at_k),
    ("Precision", precision_at_k),
    ("F1",        f1_at_k),
]

for name, fn in metrics:
    row = f"{name:<15}"
    for k in ks:
        row += f"{fn(ranked_results, qrels, k):>10.4f}"
    print(row)

print(f"\nTotal queries evaluated: {len(ranked_results)}")

# ── Save raw results to JSON ──────────────────────────────────────────────────
# Same format as tfidf_results.json and bm25_results.json from the Rust project.
# Each entry is [query_id, [doc_id, doc_id, ...]]

output_path = os.path.join(DATASET_PATH, "bert_results.json")
results_list = [[qid, doc_ids] for qid, doc_ids in ranked_results.items()]
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_list, f, indent=2)
logger.info(f"Raw results saved to: {output_path}")
