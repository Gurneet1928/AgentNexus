# pip install -qU langchain-openai langchain-groq langchain-ollama langchain-anthropic
from langchain_openai import ChatOpenAI
import streamlit as st

def lmstudio():
    try:
        model = ChatOpenAI(
                base_url="http://localhost:1234/v1",
                api_key="1234"
            )
        return model
    except Exception as e:
        st.error(f"Error initializing LM Studio: {e}")

def azureoai(
        api_key: str,
        api_version: str,
        azure_endpoint: str,
        azure_deployment: str,    
    ):
    try:
        from langchain_openai import AzureChatOpenAI
        if not all([api_key, api_version, azure_endpoint, azure_deployment]):
            raise ValueError("All parameters must be provided.")
        llm = AzureChatOpenAI(
            api_key=api_key,
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            azure_deployment=azure_deployment,
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Azure OpenAI: {e}")

def openai(
        api_key: str,
        model: str,
    ):
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing OpenAI: {e}")

def groq(
        api_key: str,
        model_name: str,
    ):
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Groq: {e}")

def ollama(
        model_name: str,    
    ):
    try:
        llm = ChatOpenAI(
            base_url="http://localhost:11434/api/chat",
            model=model_name,
            api_key="1234"
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Ollama: {e}")

def anthropic(
        api_key: str,
        model_name: str,
    ):
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            api_key=api_key,
            model=model_name,
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Anthropic: {e}")


