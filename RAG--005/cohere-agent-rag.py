import os

import dotenv
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import Chroma
from langchain.agents import AgentExecutor
from langchain_cohere.react_multi_hop.agent import create_cohere_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools.retriever import create_retriever_tool
from langchain_cohere.chat_models import ChatCohere

dotenv.load_dotenv()

# Create the Web search Tool
internet_search = TavilySearchResults()
internet_search.name = "internet_search"
internet_search.description = "Returns a list of relevant document snippets for a textual query retrieved from the internet."

class TavilySearchInput(BaseModel):
    query: str = Field(description="Query to search the internet with")

internet_search.args_schema = TavilySearchInput

# Create RAG Tool

# Set embeddings
embd = CohereEmbeddings()

# Load Docs to Index
loader = PyMuPDFLoader('./income-tax-slab.pdf') #PDF Path
data = loader.load()

#print(data[10])

# Split
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=512, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(data)

# Add to vectorstore
vectorstore = Chroma.from_documents(persist_directory='./vector',
    documents=doc_splits,
    embedding=embd,
)

vectorstore_retriever = vectorstore.as_retriever()

# Build retriever tool
vectorstore_search = create_retriever_tool(
    retriever=vectorstore_retriever,
    name="vectorstore_search",
    description="Retrieve relevant info from a vectorstore that contains documents related to Income Tax of India New and Old Regime Rules",
)

# RAG agent tool
# LLM
chat = ChatCohere(model="command-r-plus", temperature=0.3)

# Preamble
preamble = """
You are an expert who answers the user's question with the most relevant datasource.
You are equipped with an internet search tool and a special vectorstore of information about Income Tax Rules and Regulations of India.
If the query covers the topics of Income tax old and new regime India Rules and regulations then use the vectorstore search.
"""

# Prompt
prompt = ChatPromptTemplate.from_template("{input}")

# Create the ReAct agent
agent = create_cohere_react_agent(
    llm=chat,
    tools=[internet_search, vectorstore_search],
    prompt=prompt,
)

# Agent Executor
agent_executor = AgentExecutor(
    agent=agent, tools=[internet_search, vectorstore_search], verbose=True
)

# Asking Query on Current Affairs
output = agent_executor.invoke(
    {
        "input": "What is the general election schedule of India 2024?",
        "preamble": preamble,
    }
)

print(output)
print(output['output'])

# Query related to  Document
output = agent_executor.invoke(
    {
        "input": "How much deduction is required for a salary of 13lakh so that Old regime is better tahn New regime Threshold?",
        "preamble": preamble,
    }
)

print(output)
print(output['output'])

# Direct answer
output = agent_executor.invoke(
    {
        "input": "What is your name?",
        "preamble": preamble,
    }
)

print(output)
print(output['output'])