import os
import dotenv
import requests
import weaviate
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_weaviate.vectorstores import WeaviateVectorStore

dotenv.load_dotenv()

url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

# ----------------

# weaviate_client = weaviate.connect_to_local()
weaviate_client = weaviate.connect_to_wcs(cluster_url="https://my-vector-392mwjps.weaviate.network", auth_credentials=weaviate.classes.init.Auth.api_key(os.environ['WEAVIATE_API_KEY']),)
vectorstore = WeaviateVectorStore.from_documents(documents,  OpenAIEmbeddings(), client=weaviate_client)

# -------------------
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(separators=['\n', ' ', '.'], chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_text(state_of_the_union)

# ----------------------

docsearch_vector_store = WeaviateVectorStore.from_texts(
    texts,
    OpenAIEmbeddings(),
    client=weaviate_client,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
)

# ------ similarity search ------
query = "What did the president say about Justice Breyer"
docs = docsearch_vector_store.similarity_search(query, k=10)

# Print the first 100 characters of each of the 10 results.
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:100] + "...")

# ------ score ------
docs = docsearch_vector_store.similarity_search_with_score("country", k=5)
for doc in docs:
    print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")

#------- retriever -------
retriever = docsearch_vector_store.as_retriever(search_type="mmr")

template = """You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)

print(prompt)

llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

rag_chain = (
    {"context": retriever,  "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

query = "What did the president say about Justice Breyer?"
print(rag_chain.invoke(query))