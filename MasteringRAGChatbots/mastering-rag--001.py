"""
The conversation flow is a crucial component that governs when to leverage the RAG application and when to rely on the chat model.
This decision-making process comprises five distinct steps:

Question Reshaping Decision —
The first step evaluates whether the user’s question requires reshaping or can be processed as is.
This decision is made by prompting the LLM with the user’s question and relevant context.

Standalone Question Generation —
If reshaping is necessary, this step transforms the original question into a standalone query that encapsulates all the required context and information.
The LLM is prompted to generate a self-contained question that can be understood and answered without relying on external factors.

Inner Router Decision —
Once the question is reshaped into a suitable format, the inner router determines the appropriate path for obtaining a comprehensive answer.
This decision-making process involves the following steps:

Vectorstore Relevance Check:
The inner router first checks the vectorstore for relevant sources that could potentially answer the reshaped question.
If relevant sources are found, the query is forwarded to the RAG application for generating a response based on the retrieved information.

LLM Evaluation:
If no relevant sources are found in the vectorstore, the reshaped question is prompted to the LLM.
The LLM evaluates the query and determines the best path based on the information needed to provide a suitable answer.

RAG Application Route:
Despite the absence of relevant sources in the vectorstore, the LLM may still recommend utilizing the RAG application.
In such cases, the RAG application is invoked, and a "no answer" response is returned, indicating that the query cannot be satisfactorily addressed with the available information.

Chat Model Route:
If the LLM deems the chat model's capabilities sufficient to address the reshaped question, the query is processed by the chat model,
which generates a response based on the conversation history and its inherent knowledge.
"""
from typing import List, Any, Optional, Dict

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.output_parsers import YamlOutputParser
from langchain_core.callbacks import Callbacks, CallbackManagerForChainRun
from langchain_core.language_models import BaseLanguageModel
from pydantic import BaseModel

# -----------------------------------------------------------------------------------------------------------------------------------------------------

REPHRASING_PROMPT = """
# Context #
This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions. 

#########

# Objective #
Evaluate the given user question and determine if it requires reshaping according to chat history to provide necessary context and information for answering, or if it can be processed as is.

#########

# Style #
The response should be clear, concise, and in the form of a straightforward decision - either "Reshape required" or "No reshaping required".

#########

# Tone #
Professional and analytical.

#########

# Audience #
The audience is the internal system components that will act on the decision.

#########

# Response #
If the question should be rephrased return response in YAML file format:
```
    result: true
```
otherwise return in YAML file format:
```
    result: false
```

##################

# Chat History #
{chat_history}

#########

# User question #
{ question }

#########

# Your Decision in YAML format #
"""

# -----------------------------------------------------------------------------------------------------------------------------------------------------

STANDALONE_PROMPT = """
# Context #
This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions. 

#########

# Objective #
Take the original user question and chat history, and generate a new standalone question that can be understood and answered without relying on additional external information.

#########

# Style #
The reshaped standalone question should be clear, concise, and self-contained, while maintaining the intent and meaning of the original query.

#########

# Tone #
Neutral and focused on accurately capturing the essence of the original question.

#########

# Audience #
The audience is the internal system components that will act on the decision.

#########

# Response #
If the original question requires reshaping, provide a new reshaped standalone question that includes all necessary context and information to be self-contained.
If no reshaping is required, simply output the original question as is.

##################

# Chat History #
{chat_history}

#########

# User original question #
{ question }

#########

# The new Standalone question #
"""

# -----------------------------------------------------------------------------------------------------------------------------------------------------

ROUTER_DECISION_PROMPT = """
# Context #
This is part of a conversational AI system that determines whether to use a retrieval-augmented generator (RAG) or a chat model to answer user questions. 

#########

# Objective #
Evaluate the given question and decide whether the RAG application is required to provide a comprehensive answer by retrieving relevant information from a knowledge base, or if the chat model's inherent knowledge is sufficient to generate an appropriate response.

#########

# Style #
The response should be a clear and direct decision, stated concisely.

#########

# Tone #
Analytical and objective.

#########

# Audience #
The audience is the internal system components that will act on the decision.

#########

# Response #
If the question should be rephrased return response in YAML file format:
```
    result: true
```
otherwise return in YAML file format:
```
    result: false
```

##################

# Chat History #
{chat_history}

#########

# User question #
{ question }

#########

# Your Decision in YAML format #
"""


class ResultYAML(BaseModel):
    result: bool


class ConversationalRagChain(Chain):
    """Chain that encpsulate RAG application enablingnatural conversations"""
    rag_chain: Chain
    rephrasing_chain: LLMChain
    standalone_question_chain: LLMChain
    router_decision_chain: LLMChain
    yaml_output_parser: YamlOutputParser

    # input\output parameters
    input_key: str = "query"
    chat_history_key: str = "chat_history"
    output_key: str = "result"

    @property
    def input_keys(self) -> List[str]:
        """Input keys."""
        return [self.input_key, self.chat_history_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys."""
        return [self.output_key]

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "ConversationalRagChain"

    @classmethod
    def from_llm(
            cls,
            rag_chain: Chain,
            llm: BaseLanguageModel,
            callbacks: Callbacks = None,
            **kwargs: Any,
    ):
        """Initialize from LLM."""

        # create the rephrasing chain
        rephrasing_chain = LLMChain(llm=llm, prompt=REPHRASING_PROMPT, callbacks=callbacks)

        # create the standalone question chain
        standalone_question_chain = LLMChain(llm=llm, prompt=STANDALONE_PROMPT, callbacks=callbacks)

        # router decision chain
        router_decision_chain = LLMChain(llm=llm, prompt=ROUTER_DECISION_PROMPT, callbacks=callbacks)

        return cls(
            rag_chain=rag_chain,
            rephrasing_chain=rephrasing_chain,
            standalone_question_chain=standalone_question_chain,
            router_decision_chain=router_decision_chain,
            yaml_output_parser=YamlOutputParser(pydantic_object=ResultYAML),
            callbacks=callbacks,
            **kwargs,
        )

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[CallbackManagerForChainRun] = None) -> Dict[str, Any]:
        """Call the chain."""
        chat_history = inputs[self.chat_history_key]
        question = inputs[self.input_key]

        # chat history is empty, so don't rephrase the question and start a new conversation and direct to RAG chain.
        if not chat_history:
            answer = self.rag_chain.invoke({"query": question})['result'].strip()
            return {self.output_key: answer}

        # first check for question rephrasing and then check for standalone question
        needs_rephrasing = self.rephrasing_chain.invoke({"chat_history": chat_history, "question": question})[
            'text'].strip()

        # parse the llm response using the output parser
        rephrasing_decision = self.yaml_output_parser.parse(needs_rephrasing)

        # If the question need rephrasing, return the original question
        if rephrasing_decision.result:
            # Combine previous chat history and question into a single prompt to generate a standalone question.
            standalone_question = \
                self.standalone_question_chain.invoke({"chat_history": chat_history, "question": question})[
                    'text'].strip()

        # check if there are relevant sources in the vectorstore
        docs = self.rag_chain._get_docs(question)
        if docs:
            # use the RAG model
            answer = self.rag_chain.invoke({"query": question})['result'].strip()
        else:
            # send the question to inner router decision, Inner routing --> use chat model or RAG model
            routing_decision = \
                self.router_decision_chain.invoke({"chat_history": chat_history, "question": question}, )[
                    "text"].strip()

            # parse the llm response using the output parser
            routing_decision = self.yaml_output_parser.parse(routing_decision)

            if routing_decision.result:
                # Use chat model
                answer = self.llm.invoke(chat_history + [{"role": "user", "content": question}], ).content
            else:
                # get the response from the RAG chain
                answer = self.rag_chain.invoke({"query": question})['result'].strip()

        return {self.output_key: answer}
