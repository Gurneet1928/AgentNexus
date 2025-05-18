import numpy as np
import pandas as pd
import streamlit as st
from utilities.utils import listLlms, getFlow, get_thinking_message
from utilities import providers
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
providersList = {
    "LM Studio": providers.lmstudio,
    "Azure OpenAI": providers.azureoai,
    "OpenAI": providers.openai,
    "Groq": providers.groq,
    "Ollama": providers.ollama,
    "Anthropic": providers.anthropic,
}

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'provider' not in st.session_state:
    st.session_state.provider = None
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'selected_tools' not in st.session_state:
    st.session_state.selected_tools = []
if 'model_instance' not in st.session_state:
    st.session_state.model_instance = None

st.title("Agents-Ground")

@st.dialog("Model Provider Configuration", width='large')
def providerConfig():
    llmProvider = st.selectbox(
        label="Available Providers",
        options=providersList.keys(),
        placeholder="Select LLM Provider",
        index=None
    )
    st.session_state.provider = llmProvider
    try:
        if st.session_state.provider is None:
            st.error("Please select a LLM provider.")
            st.stop()
        elif llmProvider != "LM Studio" and llmProvider != "Ollama":
            if llmProvider == "Azure OpenAI":
                api_key = st.text_input("API Key", type="password")
                api_version = st.text_input("API Version")
                azure_endpoint = st.text_input("Azure Endpoint")
                azure_deployment = st.text_input("Azure Deployment")
                if st.button("Load Model"):
                    model = providersList[llmProvider](api_key, api_version, azure_endpoint, azure_deployment)
                    st.write("Model Loaded Succesfully")
            elif llmProvider != "OpenAI" or llmProvider != "Groq" or llmProvider != "Anthropic":
                api_key = st.text_input("API Key", type="password")
                model_name = st.text_input("Model Name")
                if st.button("Load Model"):
                    model = providersList[llmProvider](api_key, model_name)
                    st.write("Model Loaded Succesfully")
        elif llmProvider == "Ollama":
            import ollama
            llms = ollama.list()
            llms = {model["name"]: model for model in ollama.list()["models"]}
            if len(llms) == 0:
                st.error("No models found. Please install models using `ollama pull <model_name>`")
            chatModel = st.selectbox(
                label="Available LLMs",
                options=llms.keys(),
                placeholder="Select an LLM",
                index=None
            )
            if chatModel is not None:
                model = providersList[llmProvider](chatModel)
                st.write("Model Loaded Succesfully")
        else:
            import lmstudio as lms
            llms = listLlms()
            if len(llms.keys()) == 0:
                st.error("No models found. Please install models using `lmstudio pull <model_name>`")
                st.stop()
            chatModel = st.selectbox(
                label="Available LLMs",
                options=llms.keys(),
                placeholder="Select an LLM",
                index=None
            )
            if chatModel is not None:
                lms.llm(chatModel)
                model = providersList[llmProvider]()
                st.write("Model Loaded Succesfully")
        
        st.session_state.model_instance = model
        st.session_state.model_loaded = True
        st.rerun()
    except Exception as e:
        print(f"Error loading model: {e}")
        st.stop()

with st.sidebar:
    # if st.session_state.provider is None:
    st.button("Configure Model Provider", on_click=providerConfig, key="config_button")
    
    if st.session_state.model_instance is not None:    
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
            value=4098,
            step=10,
            help="Maximum number of tokens to generate in the output."
        )
        model = st.session_state.model_instance
        model.temperature = temperature
        model.max_tokens = max_tokens
        toolList = st.multiselect(
            label="Available Tools",
            options=toolsList.keys(),
            placeholder="Select Tools to Use",
            default=st.session_state.selected_tools
        )
        st.session_state.selected_tools = toolList

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()

if st.session_state.model_instance is not None:
    model = st.session_state.model_instance
    messages = st.container(height=500)
    for message in st.session_state.chat_history:
        messages.chat_message(message["role"]).write(message["content"])

    system_prompt = f"""
    You are a highly capable AI assistant with access to multiple specialized tools. Your goal is to solve user problems thoroughly and accurately by leveraging these tools appropriately.
    
    Available tools: [{', '.join(toolsList.keys())}]
    
    Instructions:
    1. Carefully analyze each user query to determine which tools are needed to solve it completely
    2. For complex queries, break down the problem into steps and use different tools sequentially
    3. When a problem requires multiple pieces of information, use multiple tools to gather all necessary data
    4. Combine information from different tools to provide comprehensive answers
    5. Always explain your reasoning and how you arrived at your conclusion
    6. Cite sources when you retrieve information from external tools
    
    Remember that many real-world problems require multiple tools working together. Don't hesitate to use multiple tools for a single query when appropriate.
    
    Example multi-tool workflows:
    - Research a topic using ArXiv Search, then analyze findings with Python
    - Scrape a website for data, then process that data with Python
    - Look up financial information with Yahoo Finance, then perform calculations on the results
    
    Be thorough, polite, and detail-oriented in your responses.
    """
        
    tools = [toolsList[tool] for tool in st.session_state.selected_tools]
    print(f"Tools List: {tools}")
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
    agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)
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
