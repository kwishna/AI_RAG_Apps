import datasets
import matplotlib.pyplot as plt
import pandas as pd
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from tqdm.notebook import tqdm
from transformers import AutoTokenizer
from langchain.prompts import PromptTemplate

# https://medium.com/the-ai-forum/implementing-a-flavor-of-corrective-rag-using-langchain-chromadb-zephyr-7b-beta-and-openai-30d63e222563

pd.set_option("display.max_colwidth", None)  # this will be helpful when visualizing retriever outputs

# ---------------- Load the Knowledge Source ------------------------------

ds = datasets.load_dataset("m-ric/huggingface_doc", split="train")

#  -------------- Format the datasets into Langchain Document Schema -------------
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
]
print(len(RAW_KNOWLEDGE_BASE))
print(RAW_KNOWLEDGE_BASE[1].page_content)
print(RAW_KNOWLEDGE_BASE[1].metadata)

# ---------------------------- Split the documents into chunks ---------------------

# We use a hierarchical list of separators specifically tailored for splitting Markdown documents
# This list is taken from LangChain's MarkdownTextSplitter class.
MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # the maximum number of characters in a chunk: we selected this value arbitrarily
    chunk_overlap=100,  # the number of characters to overlap between chunks
    add_start_index=True,  # If `True`, includes chunk's start index in metadata
    strip_whitespace=True,  # If `True`, strips whitespace from the start and end of every document
    separators=MARKDOWN_SEPARATORS,
)

# -------------------------------- Setup Embedding Model -------------------------------

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])

    # To get the value of the max sequence_length, we will query the underlying `SentenceTransformer` object used in the RecursiveCharacterTextSplitter.
    print(
        f"Model's maximum sequence length: {SentenceTransformer(model_name_or_path=EMBEDDING_MODEL_NAME).max_seq_length}")

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]

    # Plot the distribution of document lengths, counted as the number of tokens
    fig = pd.Series(lengths).hist()
    plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
    plt.show()
# The chunk lengths are not aligned with our limit of 512 tokens, and some documents are above the limit, thus some part of them will be lost in truncation!
# -------------- Improving Embeddings -----------------------------------

chunk_size = 512

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
    chunk_size=chunk_size,
    chunk_overlap=int(chunk_size / 10),
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)
#
docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])

print(len(docs_processed))  # 19983

# ---------------- visualize the chunk sizes we would have in tokens from a common model --------------

tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
fig = pd.Series(lengths).hist()
plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
plt.show()

# ------------------------ Setup VectorStore -----------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=True,
    model_kwargs={"device": "cpu", 'trust_remote_code': True},  # cuda
    encode_kwargs={"normalize_embeddings": True},  # set True for cosine similarity
)

# load docs into Chroma DB
KNOWLEDGE_VECTOR_DATABASE: Chroma = Chroma.from_documents(docs_processed,
                                                          embedding_model,
                                                          persist_directory="./CRAG",
                                                          collection_name="crag")

KNOWLEDGE_VECTOR_DATABASE.persist()
print(len(KNOWLEDGE_VECTOR_DATABASE.get()['documents']))
# print(collection.get(include=['embeddings', 'documents', 'metadatas']))

# Load
# KNOWLEDGE_VECTOR_DATABASE = Chroma(persist_directory="./CRAG", embedding_function=embedding_model)

# ----------------------- Check if the embeddings are retrieved based on user query -----------------

user_query = "How to create a pipeline object?"
print(f"\nStarting retrieval for {user_query=}...")

retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)

print("\n==================================Top document==================================")
print(retrieved_docs[1].page_content)

print("==================================Metadata==================================")
print(retrieved_docs[1].metadata)

# ------------------------- Setup the LLM -------------------------------

from transformers import pipeline
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline

READER_MODEL_NAME = "microsoft/Phi-3-mini-128k-instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)
#

llm = HuggingFacePipeline(pipeline=READER_LLM)

# --------------------- Create a Prompt ------------------

prompt_in_chat_format = [
    {
        "role": "system",
        "content": """Using the information contained in the context, give a comprehensive answer to the question.
        Respond only to the question asked, response should be concise and relevant to the question. 
        Provide the number of the source document when relevant. If the answer cannot be deduced from the context, do not give an answer.""",
    },
    {
        "role": "user",
        "content": """Context: {context}
        ---
        Now here is the question you need to answer.
        
        Question: {question}""",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(prompt_in_chat_format, tokenize=False, add_generation_prompt=True, model_kwargs={"device": "cpu", 'trust_remote_code': True})
print(RAG_PROMPT_TEMPLATE)

retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # we only need the text of the documents
context = "\nExtracted documents:\n"
context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

final_prompt = RAG_PROMPT_TEMPLATE.format(question="How to create a pipeline object?", context=context)

print(final_prompt)

from langchain_core.output_parsers import JsonOutputParser

answer = llm(final_prompt)
print(answer)

# ---------------- Ranker ----------------------------
from ragatouille import RAGPretrainedModel

RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
#
print("=> Reranking documents...")
question = "How to create a pipeline object?"
relevant_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=question, k=5)
relevant_docs = [doc.page_content for doc in relevant_docs]

reranked_relevant_docs = RERANKER.rerank(question, relevant_docs, k=3)

reranked_docs = [doc["content"] for doc in reranked_relevant_docs]

# Compare the documents retrieved for normal vector search and rereanker

for i, doc in enumerate(relevant_docs[:3]):
    print(f"Document {i}: {doc}")
    print("=" * 80)

for i, doc in enumerate(reranked_docs):
    print(f"Document {i}: {doc}")
    print("=" * 80)

# --------- Create a prompt

retrieved_docs_text = [doc for doc in reranked_docs]  # we only need the text of the documents
context = "\nExtracted documents:\n"
context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

final_prompt = RAG_PROMPT_TEMPLATE.format(question="How to create a pipeline object?", context=context)

print(final_prompt)

answer = llm(final_prompt)
print(answer)

# ------------ Apply corrective RAG

from langchain_openai import OpenAI

llm_openai = OpenAI(temperature=0)
#
c_prompt = PromptTemplate.from_template(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
        Provide only the binary score as a text variable with a single key 'score' and no preamble or explanation.""",
    input_variables=["question", "context"],
)
#

score_prompt = """You are a grader assessing relevance of a retrieved document to a user question. \n
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keywords related to the user question, grade it as relevant. \n
        It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n"""
#
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import PromptTemplate

response_schemas = [
    ResponseSchema(name="Score", description="score for the context query relevancy"),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
#
format_instructions = output_parser.get_format_instructions()
#
print(output_parser)
#
print(format_instructions)

template = score_prompt + "\n{format_instructions}"
print(template)
#
scoreprompt = PromptTemplate.from_template(template=template)
print(f"scoreprompt : {scoreprompt}")

# -------------- Prepared the final prompt to apply Corrective RAG ------------

question = "How to create a pipeline object?"
context = reranked_docs[0]
final_prompt = scoreprompt.format_prompt(format_instructions=format_instructions,
                                         question=question,
                                         context=context,
                                         ).text
print(final_prompt)

"""
```json
{
 "Score": string  // score for the context query relevancy
}
```
"""

# --------- Corrective RAG ----------
# Score
filtered_docs = []
grade_ = []
matched_relevant_docs = []
question = "How to create a pipeline object?"

search = "No"  # Default do not opt for web search to supplement retrieval
for d in reranked_docs:
    final_prompt = scoreprompt.format_prompt(format_instructions=format_instructions, question=question, context=d).text
    print(final_prompt)

    score = llm_openai(final_prompt)
    print(score)

    score_dict = eval(score.split("```json\n")[-1].replace("\n```", "").replace("\t", "").replace("\n", ""))
    print(score_dict)

    if score_dict['Score'] == "yes":
        matched_relevant_docs.append(d)

retrieved_docs_text = [doc for doc in matched_relevant_docs]  # we only need the text of the documents
context = "\nExtracted documents:\n"
context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

final_prompt = RAG_PROMPT_TEMPLATE.format(question="How to create a pipeline object?", context=context)

print(final_prompt)

answer = llm(final_prompt)
print(f"Response Synthesized by LLM :\n\n{answer}")
