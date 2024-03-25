# common
import os
from typing import Dict, TypedDict

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        keys: A dictionary where each key is a string.
    """

    keys: Dict[str, any]

COLOR = {
    "HEADER": "\033[95m",
    "BLUE": "\033[94m",
    "GREEN": "\033[92m",
    "RED": "\033[91m",
    "ENDC": "\033[0m",
}


# nodes

import sys, os
from operator import itemgetter
import subprocess


from langchain.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_anthropic.experimental import ChatAnthropicTools
from langchain import hub

ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")

class Nodes:
    def __init__(self, context: str, debug: bool = False):
        self.context = context
        self.debug = debug
        self.model = (
            "claude-3-opus-20240229"
        )
        self.node_map = {
            "generate": self.generate,
            "check_code_execution": self.check_code_execution,
            "finish": self.finish,
        }

    def generate(self, state: GraphState) -> GraphState:
        """
        Generate a code solution based on docs and the input question
        with optional feedback from code execution tests

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, documents, that contains retrieved documents
        """

        ## State
        state_dict = state["keys"]
        question = state_dict["question"]
        iter = state_dict["iterations"]

        ## Data model
        class code(BaseModel):
            """Code output"""

            prefix: str = Field(
                description="Description of the problem and approach"
            )
            code: str = Field(
                description="Code block not including import statements"
            )

        # LLM with tool and enforce invocation
        # llm_with_tool = ChatAnthropicTools(
        #     model="claude-3-opus-20240229",
        #     anthropic_api_key=ANTHROPIC_API_KEY,
        #     ).bind(
        #     tools=[code],
        #     tool_choice="code",
        # ).with_config(
        #     run_name="code"
        # )
            
        llm_with_tool = ChatAnthropicTools(model="claude-3-opus-20240229").bind_tools(
            tools=[code],
            tool_choice="code",
        ).with_config(
            run_name="code"
        ) 

        # Parser
        parser_tool = PydanticToolsParser(tools=[code])

        ## Prompt
        prompt = hub.pull("neuronslab/aws_cdk_engineer")

        # Chain
        chain = (
            {
                "context": lambda _: self.context,
                "question": itemgetter("question"),
                "generation": itemgetter("generation"),
                "error": itemgetter("error"),
            }
            | prompt
            | llm_with_tool
            | parser_tool
        )

        ## Generation
        if "error" in state_dict:
            print("---RE-GENERATE SOLUTION w/ ERROR FEEDBACK---")

            error = state_dict["error"]
            code_solution = state_dict["generation"]

            code_solution = chain.invoke(
                {
                    "question": question,
                    "generation": str(code_solution[0]),
                    "error": error,
                }
            )

        else:
            print("---GENERATE SOLUTION---")

            code_solution = chain.invoke(
                {
                    "question": question,
                    "generation": "",
                    "error": ""
                }
            )

        iter = iter + 1
        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "iterations": iter,
            }
        }

    def check_code_execution(self, state: GraphState) -> GraphState:
        """
        Check code block execution

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): New key added to state, error
        """

        ## State
        print("---CHECKING CODE EXECUTION---")
        state_dict = state["keys"]
        question = state_dict["question"]
        code_solution = state_dict["generation"]
        try:
            prefix = code_solution[0].prefix
        except:
            prefix = "None"
        code_block = code_solution[0].code
        iter = state_dict["iterations"]

        print(
            f"{COLOR['GREEN']}{code_block}{COLOR['ENDC']}",
            sep="\n",
        )

        import tempfile
        tmp = tempfile.NamedTemporaryFile()
        tmp.write(code_block.encode())
        tmp.flush()
        result = subprocess.run(
            f'cdk synth --validation --app "python3 {tmp.name}"',
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            )
        tmp.close()


        output, error = result.stdout, result.stderr
        if result.returncode:
            print("---CODE BLOCK CHECK: FAILED---")
            error = f"Execution error: {error}"
            print(f"Error: {error}", file=sys.stderr)
            if "error" in state_dict:
                error_prev_runs = state_dict["error"]
                error = (
                    error_prev_runs
                    + "\n --- Most recent run output and error --- \n"
                    " ------ output ------ \n"
                    + output
                    + "\n ------ error ------ \n"
                    + error
                )
        else:
            print("---CODE BLOCK CHECK: SUCCESS---")
            # No errors occurred
            error = "None"

        return {
            "keys": {
                "generation": code_solution,
                "question": question,
                "error": error,
                "prefix": prefix,
                "iterations": iter,
                "code": code_block,
            }
        }

    def finish(self, state: GraphState) -> dict:
        """
        Finish the graph

        Returns:
            dict: Final result
        """

        print("---FINISHING---")

        response = extract_response(state)

        return {"keys": {"response": response}}


def extract_response(state: GraphState) -> str:
    """
    Extract the response from the graph state

    Args:
        state (dict): The current graph state

    Returns:
        str: The response
    """

    state_dict = state["keys"]
    code_solution = state_dict["generation"][0]
    prefix = code_solution.prefix
    code = code_solution.code

    return {
        "prefix": prefix,
        "code": code,
    }


# Edges

"""Defines functions that transition our agent from one state to another."""


def enrich(graph):
    """Adds transition edges to the graph."""

    graph.add_edge("generate", "check_code_execution")
    graph.add_conditional_edges(
        "check_code_execution",
        decide_to_finish,
        {
            "finish": "finish",
            "generate": "generate",
        },
    )

    return graph


def decide_to_check_code_exec(state: GraphState) -> str:
    """
    Determines whether to test code execution, or re-try answer generation.

    Args:
    state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]

    if error == "None":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print("---DECISION: TEST CODE EXECUTION---")
        return "check_code_execution"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


def decide_to_finish(state: GraphState) -> str:
    """
    Determines whether to finish (re-try code 3 times).

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---DECIDE TO TEST CODE EXECUTION---")
    state_dict = state["keys"]
    error = state_dict["error"]
    iter = state_dict["iterations"]

    if error == "None" or iter >= 3:
        print("---DECISION: FINISH---")
        return "finish"
    else:
        print("---DECISION: RE-TRY SOLUTION---")
        return "generate"


# Agent

def construct_graph(debug=False):
    from langgraph.graph import StateGraph

    context = "" # retrieval.retrieve_docs(debug=debug)

    graph = StateGraph(GraphState)

    # attach our nodes to the graph
    graph_nodes = Nodes(context, debug=debug)
    for key, value in graph_nodes.node_map.items():
        graph.add_node(key, value)

    # construct the graph by adding edges
    graph = enrich(graph)

    # set the starting and ending nodes of the graph
    graph.set_entry_point(key="generate")
    graph.set_finish_point(key="finish")

    return graph


# App
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

#import logging

#logging.basicConfig(level=logging.DEBUG)


web_app = FastAPI(
    title="AI CodeGen Server",
    version="1.0",
    description="Generate AWS CDK code.",
)


# Set all CORS enabled origins
web_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)


def serve():
    from langchain_core.runnables import RunnableLambda
    from langserve import add_routes

    def inp(question: str) -> dict:
        return {"keys": {"question": question, "iterations": 0}}

    def out(state: dict) -> str:
        if "keys" in state:
            return state["keys"]["response"]
        elif "generate" in state:
            return extract_response(state["generate"])
        else:
            return str(state)

    graph = construct_graph(debug=False).compile()

    chain = RunnableLambda(inp) | graph | RunnableLambda(out)

    add_routes(
        web_app,
        chain,
        path="/codelangchain",
    )

    return web_app
serve()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(web_app, host="localhost", port=8000)