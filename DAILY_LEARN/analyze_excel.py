from langchain_community.document_loaders import UnstructuredExcelLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI, OpenAI
from langchain.agents import AgentExecutor, AgentType, create_openai_tools_agent, initialize_agent, load_tools
from langchain.tools.retriever import create_retriever_tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_community.vectorstores import InMemoryVectorStore
import os
import pandas as pd
import dotenv
dotenv.load_dotenv()

df = pd.read_excel(os.path.abspath("./store-sales.xlsx"), sheet_name="Orders")

# agent = create_pandas_dataframe_agent(OpenAI(temperature=0, model="gpt-4o"), df, verbose=True)
llm = ChatOpenAI(temperature=0, model="gpt-4o")

agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)

print(agent.invoke({"input": "calculate the total profits by calculating the sum of the Profit column"}))

# loader = UnstructuredExcelLoader(os.path.abspath("./DAILY_LEARN/store-sales.xlsx"))
# data = loader.load()
#
# store = InMemoryVectorStore.from_documents(
#     documents=data,
#     embedding=OpenAIEmbeddings()
# )
#
# retriever = store.as_retriever(search_kwargs="{'k': 4}")

tools = load_tools(["openweathermap-api"], llm)

executer = AgentExecutor(agent=agent, verbose=True, tools=tools)
print(executer.invoke({"input": "calculate the total profits by calculating the sum of the Profit column"}))




