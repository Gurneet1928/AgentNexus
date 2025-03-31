import lmstudio as lms
from lmstudio.sync_api import DownloadedLlm
import yaml

def listLlms() ->  dict:
    DownloadModels = lms.list_downloaded_models()
    llms = {}
    for model in DownloadModels:
        if isinstance(model, DownloadedLlm):
            llms[model.model_key] = model
    return llms