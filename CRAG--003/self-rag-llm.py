import os
import langchain
from langchain.llms import OpenAI
from langchain.chains import RAGChain
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StrOutputParser

# Set up the LLM (Large Language Model)
llm = OpenAI(model_name="text-davinci-003")

# Set up the retrieval model (e.g. a vector database)
vector_db = langchain.VectorDB("chroma")

# Define the RAG chain
rag_chain = RAGChain(
    llm=llm,
    retriever=vector_db,
    output_parser=StrOutputParser()
)

# Define the prompt templates
system_prompt = SystemMessagePromptTemplate("You are a knowledgeable assistant.")
human_prompt = HumanMessagePromptTemplate("User: {input}")

# Create a chatbot instance
chatbot = langchain.Chatbot(rag_chain, system_prompt, human_prompt)

# Test the chatbot
input_message = "What is the capital of France?"
output = chatbot.invoke(input_message)
print(output)