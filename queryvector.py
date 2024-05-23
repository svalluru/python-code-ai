from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis

redis_url = "redis://default:mypassword@localhost:6379"
index_name = "customerdataidx"
schema_name = "redis_schema.yaml"


embeddings = HuggingFaceEmbeddings()

rds = Redis.from_existing_index(
    embeddings,
    redis_url=redis_url,
    index_name=index_name,
    schema=schema_name)

query="how many orders are from EMEA Region ?"

#Make a query to the index
results =rds.similarity_search(query, k=4000, return_metadata=True)
for result in results:
    print(result)                           

#Work with a retriever
retriever = rds.as_retriever(search_type="similarity_distance_threshold", search_kwargs={"k": 4000, "distance_threshold": 2})
docs = retriever.get_relevant_documents(query)
#print(docs)
