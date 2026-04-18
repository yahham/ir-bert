import os
import logging
import pandas as pd
from dotenv import load_dotenv
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from indexMapping import indexMapping

load_dotenv()

logger = logging.getLogger(__name__)

es = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
    ca_certs=os.getenv("ES_CA_CERT"),
    request_timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)

print(es.ping())

# Prepare the data
df = pd.read_csv("datasets/myntra_products_catalog.csv").loc[:499]

print(df.head())

df.fillna("None", inplace=True)

print(df.isna().value_counts())

# Convert the relevant field to Vector using BERT model
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

df["DescriptionVector"] = df["Description"].apply(lambda x: model.encode(x))

# Create new index in Elasticsearch
if es.indices.exists(index="all_products"):
    logger.info(f"Deleting existing index")
    es.indices.delete(index="all_products")
es.indices.create(index="all_products", mappings=indexMapping)
logger.info(f"Index created")

# Ingest the data into index
record_list = df.to_dict("records")
for record in record_list:
    try:
        es.index(index="all_products", document=record, id=record["ProductID"])
    except Exception as e:
        print(e)

print(es.count(index="all_products"))
