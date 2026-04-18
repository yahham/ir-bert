import os
from dotenv import load_dotenv
from elasticsearch import Elasticsearch

load_dotenv()

es = Elasticsearch(
    os.getenv("ES_HOST"),
    basic_auth=(os.getenv("ES_USERNAME"), os.getenv("ES_PASSWORD")),
    ca_certs=os.getenv("ES_CA_CERT"),
    request_timeout=60,
    max_retries=3,
    retry_on_timeout=True,
)

print(es.ping())
