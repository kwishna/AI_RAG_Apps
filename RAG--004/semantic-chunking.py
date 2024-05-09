# https://medium.com/the-ai-forum/semantic-chunking-for-rag-f4733025d5f5

import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

loader = PyPDFLoader("./bert.pdf")
documents = loader.load()
#
print(len(documents))

print('*'*50)

# Perform Native Chunking(RecursiveCharacterTextSplitting)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
#
naive_chunks = text_splitter.split_documents(documents)
for chunk in naive_chunks[10:15]:
  print(chunk.page_content+ "\n")

print('*' * 50)

from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
embed_model = FastEmbedEmbeddings()

from groq import Groq
from langchain_groq import ChatGroq

"""
Perform Semantic Chunking

We’re going to be using the `percentile` threshold as an example today — but there’s three different strategies you could use on Semantic Chunking):

- `percentile` (default) — In this method, all differences between sentences are calculated, and then any difference greater than the X percentile is split.

- `standard_deviation` — In this method, any difference greater than X standard deviations is split.

- `interquartile` — In this method, the interquartile distance is used to split chunks.
"""
#
from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

semantic_chunker = SemanticChunker(embed_model, breakpoint_threshold_type="percentile")
semantic_chunks = semantic_chunker.create_documents([d.page_content for d in documents])

for semantic_chunk in semantic_chunks:
  if "Effect of Pre-training Tasks" in semantic_chunk.page_content:
    print(semantic_chunk.page_content)
    print(len(semantic_chunk.page_content))

print('*' * 50)

semantic_chunk_vectorstore = Chroma.from_documents(semantic_chunks, embedding=embed_model)
#
# ------------------- Instantiate Retrieval Step --------------------
#
semantic_chunk_retriever = semantic_chunk_vectorstore.as_retriever(search_kwargs={"k" : 1})
semantic_chunk_retriever.invoke("Describe the Feature-based Approach with BERT?")
#
# --------------------- Instantiate Augmentation Step(for content Augmentation) ---------------

rag_template = """
Use the following context to answer the user's query. If you cannot answer, please respond with 'I don't know'.

User's Query:
{question}

Context:
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(rag_template)
chat_model = ChatGroq(model_name="mixtral-8x7b-32768", temperature=0, api_key=os.environ["GROQ_API_KEY"])

# --------------- Creating a RAG Pipeline Utilizing Semantic Chunking -----------------

semantic_rag_chain = (
    {"context" : semantic_chunk_retriever, "question" : RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

# ----------------- Running the RAG Pipeline -----------------

print(semantic_rag_chain.invoke("Describe the Feature-based Approach with BERT?"))
print(semantic_rag_chain.invoke("What is SQuADv2.0?"))
print(semantic_rag_chain.invoke("What is the purpose of Ablation Studies?"))

# --------------------- Implement a RAG pipeline using Naive Chunking Strategy -----------------------------

naive_chunk_vectorstore = Chroma.from_documents(naive_chunks, embedding=embed_model)
naive_chunk_retriever = naive_chunk_vectorstore.as_retriever(search_kwargs={"k" : 5})
naive_rag_chain = (
    {"context" : naive_chunk_retriever, "question" : RunnablePassthrough()}
    | rag_prompt
    | chat_model
    | StrOutputParser()
)

# -------------------------- Ragas Assessment Comparison for Semantic Chunker ----------------------
# -------------------------- split documents using RecursiveCharacterTextSplitter ----------------------

synthetic_data_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
    length_function=len,
    is_separator_regex=False
)
synthetic_data_chunks = synthetic_data_splitter.create_documents([d.page_content for d in documents])
print(len(synthetic_data_chunks))

"""
Create the Following Datasets

Questions — synthetically generated (grogq-mixtral-8x7b-32768)
Contexts — created above(Synthetic data chunks)
Ground Truths — synthetically generated (grogq-mixtral-8x7b-32768)
Answers — generated from our Semantic RAG Chain
"""

questions = []
ground_truths_semantic = []
contexts = []
answers = []

question_prompt = """\
You are a teacher preparing a test. Please create a question that can be answered by referencing the following context.

Context:
{context}
"""

question_prompt = ChatPromptTemplate.from_template(question_prompt)

ground_truth_prompt = """\
Use the following context and question to answer this question using *only* the provided context.

Question:
{question}

Context:
{context}
"""

ground_truth_prompt = ChatPromptTemplate.from_template(ground_truth_prompt)

question_chain = question_prompt | chat_model | StrOutputParser()
ground_truth_chain = ground_truth_prompt | chat_model | StrOutputParser()

for chunk in synthetic_data_chunks[10:20]:
  questions.append(question_chain.invoke({"context" : chunk.page_content}))
  contexts.append([chunk.page_content])
  ground_truths_semantic.append(ground_truth_chain.invoke({"question" : questions[-1], "context" : contexts[-1]}))
  answers.append(semantic_rag_chain.invoke(questions[-1]))

from datasets import load_dataset, Dataset

qagc_list = []

for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths_semantic):
  qagc_list.append({
      "question" : question,
      "answer" : answer,
      "contexts" : context,
      "ground_truth" : ground_truth
  })

eval_dataset = Dataset.from_list(qagc_list)
print(eval_dataset) #  eval_dataset

###########################RESPONSE###########################
# Dataset({
#     features: ['question', 'answer', 'contexts', 'ground_truth'],
#     num_rows: 10
# })

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
)

#
from ragas import evaluate

result = evaluate(
    eval_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
     llm=chat_model,
    embeddings=embed_model,
    raise_exceptions=False
)

# groq.RateLimitError: Error code: 429 - {'error': {'message': 'Rate limit reached for model `mixtral-8x7b-32768` in organization `org_01htsyxttnebyt0av6tmfn1fy6` on tokens per minute (TPM): Limit 4500, Used 3867, Requested ~1679. Please try again in 13.940333333s. Visit https://console.groq.com/docs/rate-limits for more information.', 'type': 'tokens', 'code': 'rate_limit_exceeded'}}
import openai
from ragas import evaluate

result = evaluate(
    eval_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)
print(result) #   result

#Extract the details into a dataframe
results_df = result.to_pandas()
results_df

# ----------------------------------- Ragas Assessment Comparison for Naive Chunker ----------------------

import tqdm
questions = []
ground_truths_semantic = []
contexts = []
answers = []
for chunk in tqdm.tqdm(synthetic_data_chunks[10:20]):
  questions.append(question_chain.invoke({"context" : chunk.page_content}))
  contexts.append([chunk.page_content])
  ground_truths_semantic.append(ground_truth_chain.invoke({"question" : questions[-1], "context" : contexts[-1]}))
  answers.append(naive_rag_chain.invoke(questions[-1]))


# -------------- Formulate naive chunking evaluation dataset

qagc_list = []

for question, answer, context, ground_truth in zip(questions, answers, contexts, ground_truths_semantic):
  qagc_list.append({
      "question" : question,
      "answer" : answer,
      "contexts" : context,
      "ground_truth" : ground_truth
  })

naive_eval_dataset = Dataset.from_list(qagc_list)
print(naive_eval_dataset) # naive_eval_dataset

############################RESPONSE########################
# Dataset({
#     features: ['question', 'answer', 'contexts', 'ground_truth'],
#     num_rows: 10
# })

naive_result = evaluate(
    naive_eval_dataset,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
    ],
)
#
print(naive_result) #  naive_result
############################RESPONSE#######################
# {'context_precision': 1.0000, 'faithfulness': 0.9500, 'answer_relevancy': 0.9182, 'context_recall': 1.0000}

naive_results_df = naive_result.to_pandas()
print(naive_results_df)

###############################RESPONSE #######################
# {'context_precision': 1.0000, 'faithfulness': 0.9500, 'answer_relevancy': 0.9182, 'context_recall': 1.0000}