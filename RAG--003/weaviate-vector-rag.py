import dotenv
import weaviate
import requests
from weaviate.embedded import EmbeddedOptions
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Weaviate
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.document_loaders import TextLoader
from langchain_weaviate.vectorstores import WeaviateVectorStore

dotenv.load_dotenv()

url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

# text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
# chunks = text_splitter.split_documents(documents)
#
# client = weaviate.Client(
#   embedded_options = EmbeddedOptions()
# )
#
# vectorstore = Weaviate.from_documents(
#     client = client,
#     documents = chunks,
#     embedding = OpenAIEmbeddings(),
#     by_text = False
# )
#
# retriever = vectorstore.as_retriever()

weaviate_client = weaviate.connect_to_local()
vectorstore = WeaviateVectorStore.from_documents(documents,  OpenAIEmbeddings(), client=weaviate_client)

# ------ similarity search ------
query = "What did the president say about Justice Breyer"
docs = vectorstore.similarity_search(query, k=10)
# Print the first 100 characters of each result
for i, doc in enumerate(docs):
    print(f"\nDocument {i+1}:")
    print(doc.page_content[:100] + "...")

# ------ score ------
docs = vectorstore.similarity_search_with_score("country", k=5)
for doc in docs:
    print(f"{doc[1]:.3f}", ":", doc[0].page_content[:100] + "...")

#------ retriever ------
retriever = vectorstore.as_retriever(search_type="mmr")

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

query = "What did the president say about Justice Breyer"
rag_chain.invoke(query)