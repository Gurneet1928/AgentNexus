import numpy as np
import pandas as pd
import streamlit as st
from utilities.utils import getFlow, get_thinking_message, read_yaml
from utilities import providers
from tools import toolsHub
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
import streamlit_themes as st_themes
from pathlib import Path

modelConfig = {}
model = None
toolsList = {
    "pythonRepl": toolsHub.python_repl,
    "getWord": toolsHub.get_word_length,
    "Website Scapper": toolsHub.scrapSite,
    "ArXiv Search": toolsHub.searchArXiv,
    "Sematic Scholar": toolsHub.sematicScholar,
    "Yahoo Finance": toolsHub.yahooFinance,
}
providersList = {
    "LM Studio": providers.lmstudio,
    "Azure OpenAI": providers.azureoai,
    "OpenAI": providers.openai,
    "Groq": providers.groq,
    # "Ollama": providers.ollama,
    "Anthropic": providers.anthropic,
    "Google": providers.google,
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
if "visualization" not in st.session_state:
    st.session_state.visualization = {}

configFilePath = "config.yaml"
content = read_yaml(Path(configFilePath))

# Set page configuration
st.set_page_config(
    page_title="AgentNexus: AI Playground",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/yourusername/Agents-Ground',
        'Report a bug': 'https://github.com/yourusername/Agents-Ground/issues',
        'About': 'AgentNexus: Orchestrating multiple AI agents to solve complex problems'
    }
)

# st.header("AgentNexus: AI Playground")
st.markdown("<h1 style='text-align: center;'>AgentNexus: AI Playground</h1>", unsafe_allow_html=True)

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
            elif llmProvider != "OpenAI" or llmProvider != "Groq" or llmProvider != "Anthropic"  or llmProvider != "Google":
                api_key = st.text_input("API Key", type="password")
                model_name = st.text_input("Model Name")
                if st.button("Load Model"):
                    model = providersList[llmProvider](api_key, model_name)
                    st.write("Model Loaded Succesfully")
        elif llmProvider == "Ollama":
            import ollama
            llms = ollama.list()
            print(llms)
            llms = {model["model"]: model for model in ollama.list()["models"]}
            if len(llms) == 0:
                st.error("No models found. Please install models using `ollama pull <model_name>`")
            chatModel = st.selectbox(
                label="Available LLMs",
                options=llms.keys(),
                placeholder="Select an LLM",
                index=Nones
            )
            if chatModel is not None:
                ollama.chat(chatModel)
                model = providersList[llmProvider](chatModel)
                st.write("Model Loaded Succesfully")
        else:
            import lmstudio as lms
            llms = providers.listLlms()
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
        model = st.session_state.model_instance
        model.temperature = temperature
        if st.session_state.provider != "Google":
            max_tokens = st.slider(
                label="Max Tokens",
                min_value=1,
                max_value=8192,
                value=4098,
                step=10,
                help="Maximum number of tokens to generate in the output."
            )
            model.max_tokens = max_tokens
        else:
            st.write("Google Gemini models do not support max tokens limit.")
        toolList = st.multiselect(
            label="Available Tools",
            options=toolsList.keys(),
            placeholder="Select Tools to Use",
            default=st.session_state.selected_tools
        )
        st.session_state.selected_tools = toolList

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.visualization = {}     
            st.rerun()

        # st_themes.preset_theme_widget()

if st.session_state.model_instance is not None:
    model = st.session_state.model_instance
    messages = st.container(height=500)
    for message in st.session_state.chat_history:
        messages.chat_message(message["role"]).write(message["content"])
    if st.session_state.visualization != {}:
        messages.button(
            "Visualize reasoning", 
            on_click=getFlow,
            args=(st.session_state.visualization["input"], st.session_state.visualization["output"], st.session_state.visualization["intermediate_steps"],), 
            key=f"viz_button_{len(st.session_state.chat_history)}"
        )

    system_prompt = content["system_prompt"].format(availableTools=", ".join(toolsList.keys()))
        
    tools = [toolsList[tool] for tool in st.session_state.selected_tools]
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
        try:
            with assistant_placeholder:
                with st.spinner(get_thinking_message(), show_time=True):
                    response = agent_executor.invoke({"input": msg})
                
            # with assistant_placeholder:
                st.session_state.chat_history.append({"role": "assistant", "content": response["output"]})
                if "intermediate_steps" in response:
                    st.session_state.visualization = {
                        "input": msg,
                        "output": response["output"],
                        "intermediate_steps": response["intermediate_steps"]
                    }
                    messages.button(
                        "Visualize reasoning", 
                        on_click=getFlow,
                        args=(msg, response["output"], response["intermediate_steps"],), 
                        key=f"viz_button_{len(st.session_state.chat_history)}"
                    )
                st.rerun()
        except Exception as e:
            with assistant_placeholder:
                st.session_state.chat_history.append({"role": "assistant", "content": f"Error: {e}"})
                messages.chat_message("assistant").write(f"Ooooppppss!! Looks like we encountered an Error. Here is the error Description to help you: \n{e}")   
                st.rerun()        
else:
    st.error("Model not loaded. Please select a model from the sidebar.")
