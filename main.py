import numpy as np
import pandas as pd
import streamlit as st
import lmstudio as lms
from utilities.utils import listLlms
from tools import toolsHub 

llms = listLlms()
modelConfig = {}
model = None
tools = []

st.title("Agents-Ground")

with st.sidebar:
    chatModel = st.selectbox(
        label="Available LLMs",
        options=llms.keys(),
        placeholder="Select an LLM"
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
        model = lms.llm(chatModel)
        modelConfig = {
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        st.write("Model Loaded Succesfully")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    toolList = st.selectbox(
        label="Available Tools",
        options=dir(toolsHub),
        placeholder="Select an LLM"
    )


if model is not None:
    messages = st.container(height=600)
    if msg := st.chat_input(placeholder="Message to send to LLM"):
        messages.chat_message("user").write(msg)
        response = model.respond(
            msg,
            config=modelConfig
        )
        messages.chat_message("assistant").write(response.content)
else:
    st.error("Model not loaded. Please select a model from the sidebar.")
