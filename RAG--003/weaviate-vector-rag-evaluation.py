import os
import dotenv
import requests
import weaviate
from datasets import Dataset
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_weaviate.vectorstores import WeaviateVectorStore
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
)

dotenv.load_dotenv()

url = "https://raw.githubusercontent.com/langchain-ai/langchain/master/docs/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
    f.write(res.text)

loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

# ----------------

# weaviate_client = weaviate.connect_to_local()
weaviate_client = weaviate.connect_to_wcs(
    cluster_url="https://my-vector-392mwjps.weaviate.network",
    auth_credentials=weaviate.classes.init.Auth.api_key(os.environ['WEAVIATE_API_KEY'])
)

# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2');

vectorstore = WeaviateVectorStore.from_documents(documents, embeddings, client=weaviate_client)

# -------------------
with open("state_of_the_union.txt") as f:
    state_of_the_union = f.read()

text_splitter = RecursiveCharacterTextSplitter(separators=['\n', ' ', '.'], chunk_size=500, chunk_overlap=20)
texts = text_splitter.split_text(state_of_the_union)

# ----------------------

docsearch_vector_store = WeaviateVectorStore.from_texts(
    texts,
    embeddings,
    client=weaviate_client,
    metadatas=[{"source": f"{i}-pl"} for i in range(len(texts))],
)

# ------ similarity search ------
query = "What did the president say about Justice Breyer"
docs = docsearch_vector_store.similarity_search(query, k=10)

# Print the first 100 characters of each of the 10 results.
for i, doc in enumerate(docs):
    print(f"\nDocument {i + 1}:")
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
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

query = "What did the president say about Justice Breyer?"
print(rag_chain.invoke(query))

# -----------------------------------------------
# RAG Evaluation

questions = ["What did the president say about Justice Breyer?",
             "What did the president say about Intel's CEO?",
             "What did the president say about gun violence?",
             ]
ground_truths = [[
                     "The president said that Justice Breyer has dedicated his life to serve the country and thanked him for his service."],
                 ["The president said that Pat Gelsinger is ready to increase Intel's investment to $100 billion."],
                 ["The president asked Congress to pass proven measures to reduce gun violence."]]
answers = []
contexts = []

# Inference
for query in questions:
    answers.append(rag_chain.invoke(query))
    contexts.append([docs.page_content for docs in retriever.invoke(query)])

# To dict
data = {
    "question": questions,
    "answer": answers,
    "contexts": contexts,
    "ground_truths": ground_truths
}

# Convert dict to dataset
dataset = Dataset.from_dict(data)

result = evaluate(
    dataset=dataset,
    metrics=[
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ],
)

df = result.to_pandas()
df.to_csv(path_or_buf="./evaluated.csv")

"""
Evaluation Metrics
RAGAs provide you with a few metrics to evaluate a RAG pipeline component-wise as well as end-to-end.

On a component level, RAGAs provides you with metrics to evaluate the retrieval component (context_relevancy and context_recall) and the generative component (faithfulness and answer_relevancy) separately [2]:

Context precision measures the signal-to-noise ratio of the retrieved context. This metric is computed using the question and the contexts.
Context recall measures if all the relevant information required to answer the question was retrieved. This metric is computed based on the ground_truth
(this is the only metric in the framework that relies on human-annotated ground truth labels) and the contexts.

Faithfulness measures the factual accuracy of the generated answer.

The number of correct statements from the given contexts is divided by the total number of statements in the generated answer.
This metric uses the question, contextsand the answer.

Answer relevancy measures how relevant the generated answer is to the question.
This metric is computed using the question and the answer.
For example, the answer “France is in western Europe.” to the question “Where is France and what is it’s capital?” would achieve a low answer relevancy because it only answers half of the question.

context_relevancy (signal-to-noise ratio of the retrieved context): While the LLM judges all of the context as relevant for the last question,
it also judges that most of the retrieved context for the second question is irrelevant.
Depending on this metric, you could experiment with different numbers of retrieved contexts to reduce the noise.

context_recall (if all the relevant information required to answer the question was retrieved): The LLMs evaluate that the retrieved contexts contain the relevant information required to answer the questions correctly.

faithfulness (factual accuracy of the generated answer): While the LLM judges that the first and last questions are answered correctly, the answer to the second question,
which wrongly states that the president did not mention Intel’s CEO, is judged with a faithfulness of 0.5.

answer_relevancy (how relevant is the generated answer to the question): All of the generated answers are judged as fairly relevant to the questions.

https://towardsdatascience.com/evaluating-rag-applications-with-ragas-81d67b0ee31a#836f
"""
