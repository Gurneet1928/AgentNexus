import numpy as np
import pandas as pd
import streamlit as st
import lmstudio as lms
from utilities.utils import listLlms, getFlow, get_thinking_message
from tools.toolsHub import get_word_length, python_repl, scrapSite, searchArXiv, sematicScholar, yahooFinance
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, load_tools, tool
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_experimental.utilities import PythonREPL
# prompt = hub.pull("hwchase17/react")


llms = listLlms()
modelConfig = {}
model = None
toolsList = {
    "pythonRepl": python_repl,
    "getWord": get_word_length,
    "Website Scapper": scrapSite,
    "ArXiv Search": searchArXiv,
    "Sematic Scholar": sematicScholar,
    "Yahoo Finance": yahooFinance,
}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Agents-Ground")

with st.sidebar:
    chatModel = st.selectbox(
        label="Available LLMs",
        options=llms.keys(),
        placeholder="Select an LLM",
        index=None
        
    )
    temperature = st.slider(
        label="Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
        help="Higher values make the output more random."
    )
    max_tokens = st.slider(
        label="Max Tokens",
        min_value=1,
        max_value=8192,
        value=512,
        step=10,
        help="Maximum number of tokens to generate in the output."
    )
    try:
        if chatModel is not None:
            lms.llm(chatModel)
            modelConfig = {
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            model = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="1234",
                temperature=modelConfig["temperature"],
                max_tokens=modelConfig["max_tokens"],
                verbose=True,
            )
            st.write("Model Loaded Succesfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    toolList = st.multiselect(
        label="Available Tools",
        options=toolsList.keys(),
        placeholder="Select Tools to Use"
    )

if model is not None:
    messages = st.container(height=500)
    for message in st.session_state.chat_history:
        messages.chat_message(message["role"]).write(message["content"])

    system_prompt = f"""
    You are very powerful assistant. And must resolve user queries polietly. 
    You have access to the following tools: [{', '.join(toolsList)}].
    After receiving the user query, you must think about what to do and then use the tools to get the answer.
    After receiving the answer, you must think about how to respond to the user.
    You are free to combine multiple tools to get the final answer or execute the queries.
    """,
        
    tools = [toolsList[tool] for tool in toolList]
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"{system_prompt}",
            ),
            ("user", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    llm_with_tools = model.bind_tools(tools)

    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                x["intermediate_steps"]
            ),
        }
        | prompt
        | llm_with_tools
        | OpenAIToolsAgentOutputParser()
    )
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
    if msg := st.chat_input(placeholder="Message to send to LLM"):
        st.session_state.chat_history.append({"role": "user", "content": msg})
        messages.chat_message("user").write(msg)
        assistant_placeholder = messages.chat_message("assistant")
    
        with assistant_placeholder:
            with st.spinner(get_thinking_message(), show_time=True):
                response = agent_executor.invoke({"input": msg})
            
        st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
        messages.chat_message("assistant").write(response["output"])
        if "intermediate_steps" in response:
            messages.button(
                "Visualize reasoning", 
                on_click=getFlow,
                args=(msg, response["output"], response["intermediate_steps"],), 
                key=f"viz_button_{len(st.session_state.chat_history)}"
            )
                 
else:
    st.error("Model not loaded. Please select a model from the sidebar.")
