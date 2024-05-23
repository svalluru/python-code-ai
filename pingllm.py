from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.redis import Redis
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.llms import Ollama
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

redis_url = "redis://default:mypassword@localhost:6379"
index_name = "customerdataidx"
schema_name = "redis_schema.yaml"


embeddings = HuggingFaceEmbeddings()

rds = Redis.from_existing_index(
    embeddings,
    redis_url=redis_url,
    index_name=index_name,
    schema=schema_name)



#Work with a retriever
retriever = rds.as_retriever(search_type="similarity_distance_threshold", search_kwargs={"k": 4000, "distance_threshold": 2})

print(retriever)

retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

llm = Ollama(
    base_url="http://localhost:11434",
    model="mistral",
    callbacks=[StreamingStdOutCallbackHandler()])  


combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
#print("###########")  
#print(combine_docs_chain)
#print("###########")  

retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
#print(retrieval_chain)


#response = retrieval_chain.invoke({"input": "find total population in Tokyo in years 1930 and 1987?"})

#response = retrieval_chain.invoke({"input": "find total population in Toronto and Tokyo ?"})

response = retrieval_chain.invoke({"input": "how many orders are created for Product Alcehmy ?"})

#response = retrieval_chain.invoke({"input": "what is OpenShift AI ?"})

print(response['answer'])
print(response['context'])

