from langchain_openai import ChatOpenAI
import streamlit as st
import lmstudio as lms
from lmstudio.sync_api import DownloadedLlm

def listLlms() ->  dict:
    """
    List all available LLMs in the LM Studio.
    Args:
        None
    Raises:
        Exception: If there is an error listing the LLMs
    Returns:
        dict: A dictionary of LLMs with their model keys as keys and DownloadedLlm objects as values
    """
    try:
        DownloadModels = lms.list_downloaded_models()
        llms = {}
        for model in DownloadModels:
            if isinstance(model, DownloadedLlm):
                llms[model.model_key] = model
        return llms
    except Exception as e:
        st.error(f"Error listing LLMs: {e}")

def lmstudio():
    """
    Initialize the LM Studio model.
    Args:
        None
    Raises:
        Exception: If there is an error initializing the LM Studio model
    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class from the langchain_openai module
    """
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
    """
    Initialize the Azure OpenAI model.
    Args:
        api_key (str): The API key for Azure OpenAI
        api_version (str): The API version for Azure OpenAI
        azure_endpoint (str): The Azure endpoint URL
        azure_deployment (str): The Azure deployment name
    Raises:
        Exception: If there is an error initializing the Azure OpenAI model
    Returns:
        AzureChatOpenAI: An instance of the AzureChatOpenAI class from the langchain_openai module
    """
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
    """
    Initialize the OpenAI model.
    Args:
        api_key (str): The API key for OpenAI
        model (str): The model name for OpenAI
    Raises:
        Exception: If there is an error initializing the OpenAI model
    Returns:
        ChatOpenAI: An instance of the ChatOpenAI class from the langchain_openai module
    """
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
    """
    Initialize the Groq model.
    Args:
        api_key (str): The API key for Groq
        model_name (str): The model name for Groq
    Raises:
        Exception: If there is an error initializing the Groq model
    Returns:
        ChatGroq: An instance of the ChatGroq class from the langchain_groq module
    """
    try:
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            api_key=api_key,
            model=model_name,
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Groq: {e}")

## IN DEVELOPMENT ##
def _ollama(
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
    """
    Initialize the Anthropic model.
    Args:
        api_key (str): The API key for Anthropic
        model_name (str): The model name for Anthropic
    Raises:
        Exception: If there is an error initializing the Anthropic model
    Returns:
        ChatAnthropic: An instance of the ChatAnthropic class from the langchain_anthropic module
    """
    try:
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(
            api_key=api_key,
            model=model_name,
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Anthropic: {e}")

def google(
        api_key: str,
        model_name: str,
    ):
    """
    Initialize the Google GenerativeAI model.
    Args:
        api_key (str): The API key for Google GenerativeAI
        model_name (str): The model name for Google GenerativeAI
    Raises:
        Exception: If there is an error initializing the Google GenerativeAI model
    Returns:
        ChatGoogleGenerativeAI: An instance of the ChatGoogleGenerativeAI class from the langchain_google_genai module
    """
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model=model_name,
        )
        return llm
    except Exception as e:
        st.error(f"Error initializing Google GenAI Model: {e}")

