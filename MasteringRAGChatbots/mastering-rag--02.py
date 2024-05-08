from typing import Dict, Any, Optional, List, Mapping, Union

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain_core.callbacks import CallbackManagerForChainRun
from langchain_core.language_models import LLM
from llama_index.core import PromptTemplate
from pydantic import Extra
from semantic_router import RouteLayer, Route
from transformers.models.pop2piano.convert_pop2piano_weights_to_hf import encoder


class DummyLLM(LLM):
    """Dummy LLM for hardcoded responses."""

    responses: List[str]
    i: int = 0

    @property
    def _llm_type(self) -> str:
        return "dummy-llm"

    def _call(
            self,
            prompt: str,
            stop: Optional[List[str]] = None,
            run_manager: Optional[CallbackManagerForLLMRun] = None,
            **kwargs: Any,
    ) -> str:
        response = self.responses[self.i]
        if self.i < len(self.responses) - 1:
            self.i += 1
        else:
            self.i = 0
        return response

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"responses": self.responses}

class ExtendedRoute(Route):
    """This class extends the Route class to include a dummy llm for testing purposes."""
    responses: Optional[List[str]]
    langchain_llm: Optional[Union[DummyLLM, LLM]]

    def __init__(self, config: Dict[str, Any]):
        # initialize parent
        super().__init__(**config)

        # extract relevance data from config
        self.responses = config.get('responses', [])
        if self.responses:
            # initialize dummy llm with pre-defined responses
            self.langchain_llm = DummyLLM(responses=self.responses)

class SemanticRouter(Chain):
    """Semantic Router Chain."""
    rag_chain: Chain
    router_layer: RouteLayer
    routes: Dict[str, ExtendedRoute]
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Input keys.

        :meta private:
        """
        return [self.input_key, self.chat_history_key]

    @property
    def output_keys(self) -> List[str]:
        """Output keys.

        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    @classmethod
    def from_llm(
            cls,
            rag_chain: Chain = None,
            router_layer: Optional[RouteLayer] = None
    ):
        """Initialize from LLM."""
        return cls(default_chain=rag_chain, router_layer=router_layer,
                   routes={route.name: route for route in router_layer.routes})

    @property
    def _chain_type(self) -> str:
        """Return the chain type."""
        return "semantic router chain"

    def _call(
            self,
            inputs: Dict[str, Any],
            run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run chain on input query."""
        query = inputs[self.input_key]

        # send user query to the router layer
        route_choice = self.router_layer(query)

        # check if a route as been found
        if route_choice.name:
            # execute the route llm to get a response
            response = self.routes[route_choice.name].langchain_llm.invoke(query)
        else:
            # if no route is found, use the rag chain to answer user question
            response = self.default_chain.invoke({"query": query})[self.output_key]
        return {self.output_key: response}



router = RouteLayer(encoder=encoder, routes=list(routes.values()))
router("I need to write a Python script for web scraping. Can you assist?")

# creates a dummy chain for demonstration purposes
dummy_rag_chain = LLMChain(llm=DummyLLM(responses=["Hi, I'm a Dummy RAG Chain :)"]), prompt=PromptTemplate.from_template(""))

# creating the semantic router chain
chain = SemanticRouter.from_llm(rag_chain=dummy_rag_chain, router_layer=router)
var = chain.invoke("What's your take on investing in the healthcare sector?")['result']
print(var)