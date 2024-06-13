import json
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import requests
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.embeddings.base import Embeddings
from langchain.llms.base import LLM
from langchain_community.llms.utils import enforce_stop_tokens
from langchain.utils import get_from_dict_or_env
from pydantic.v1 import Extra, root_validator
import logging
#from telemetry import TelemetryService

logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "llama2-7b-chat-gpu"

VALID_TASKS = (
    "text-generation",
    "text2text-generation",
    "summarization",
    "question-answering",
)

DEFAULT_CONFIG = {
    "top_k": 0,
    "top_p": 0.15,
    "temperature": 0.1,
    "repetition_penalty": 1.1,
    "max_new_tokens": 64,
}


def get_datatype(value: Any) -> Tuple[str, str]:
    """Returns a tuple (datatype, content_type) for the value."""

    if isinstance(value, bool):
        return "BOOL", ""
    elif isinstance(value, float):
        return "FP64", ""
    elif isinstance(value, int):
        return "INT64", ""
    elif isinstance(value, str):
        return "BYTES", "str"
    else:
        return TypeError("Unsupported datatype!")


def encode_request(payload: Dict) -> Dict[str, Any]:
    inputs = []
    for name, value in payload.items():
        datatype, content_type = get_datatype(value)
        input = {
            "name": name,
            "shape": [1],  # it's a single value: a bool, an int, a float, a string.
            "datatype": datatype,
            "data": [value]
        }
        # required for MLServer
        if content_type:
            input.update({"parameters": {"content_type": "str"}})
        inputs.append(input)

    return dict(inputs=inputs)


class InferenceApi:
    """Client to configure requests and make calls to the Seldon V2 API."""
    def __init__(
        self,
        repo_id: str,
        task: Optional[str] = None,
        url: Optional[str] = None,
        trace: Optional[str] = None,
        prompt_id: Optional[str] = None,
    ):
        """Inits headers and API call information."""
        self.headers = {
            "Content-Type": "application/json",
            "Seldon-Model": repo_id,
        }
        self.task = task
        self.session = requests.Session()
        self.session.headers = self.headers
        self.api_url = f"{url}/v2/models/model/infer"
        #self.telemetry_service = TelemetryService(logger, trace)
        self.prompt_id = prompt_id

    def __call__(
        self,
        inputs: Optional[Union[str, Dict, List[str], List[List[str]]]] = None,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        raw_response: bool = False,
        ctx: Optional[Any] = None,
    ) -> Any:
        """Make a call to the inference API."""
        if self.task == "question-answering":
            request = {"question": inputs[0], "context": inputs[1]}
        else:
            request = {"array_inputs": inputs}
        if params is None:
            params = {}
        request.update(params)
        payload = encode_request(request)
        logger.info(f"seldon input payload: {payload}")
        request.update({"prompt_id": self.prompt_id})
        #ctx = self.telemetry_service.addTelemetry(None, "pyclient_seldon_request", ctx, request_payload=request)
        response = self.session.post(self.api_url, json=payload, data=data, headers={"Content-Type": "application/json"})
        #self.telemetry_service.addTelemetry(None, "pyclient_seldon_response", ctx, response_payload=response.json())
        print(response.json())
        response.raise_for_status()

        logger.info(f"seldon model response: {response.json()}")

        if raw_response:
            return response

        content_type = response.headers.get("Content-Type") or ""
        if content_type == "application/json":
            return response.json()
        if content_type == "text/plain":
            return response.text
        raise NotImplementedError(
            f"{content_type} output type is not implemented yet.  You can pass"
            " `raw_response=True` to get the raw `Response` object and parse the"
            " output yourself."
        )


class SeldonCore(LLM, Embeddings):
    """Seldon Core Endpoint models.

    Example:
        .. code-block:: python
            from llm_seldon.langchain import SeldonCore

            endpoint_url = (
                    "http://0.0.0.0:9000"
            )
            llm = SeldonCore(
                repo_id="gpt2",
                endpoint_url=endpoint_url,
                model_kwargs={
                    "temperature": 0.1,
                    "max_length": 128,
                    "top_p": 0.15,
                    "top_k": 0,
                    "repetition_penalty": 1.1,
                }
            )
    """

    client: Any
    repo_id: str = DEFAULT_REPO_ID
    """Model name to use"""
    task: Optional[str] = None
    """Task to call the model with.
    Should be a task that returns `generated_text` or `summary_text`."""
    model_kwargs: Optional[dict] = None
    """key word arguments to pass to the model."""
    endpoint_url: Optional[str] = None
    ctx: Optional[Any] = None
    trace: Optional[Any] = None
    prompt_id: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""
        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate environment."""
        api_host = get_from_dict_or_env(
            values, "endpoint_url", "SELDON_ENDPOINT_URL"
        )
        repo_id = values["repo_id"]
        client = InferenceApi(repo_id, values.get('task'), api_host, values["trace"], values["prompt_id"])
        values['client'] = client
        return values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{
                "repo_id": self.repo_id,
                "task": self.task
            },
            **{
                "model_kwargs": _model_kwargs
            },
        }

    @property
    def _llm_type(self) -> str:
        return "seldon"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ):
        """Call out the Seldon API moder inference endpoint."""
        _model_kwargs = self.model_kwargs or {}
        params = {**_model_kwargs, **kwargs}
        logger.info(f"model params: {params}")
        response = self.client(inputs=prompt, params=params, ctx={"Content-Type": "application/json"})
        if "error" in response:
            raise ValueError(
                f"Error raised by inference API: {response['error']}"
            )
        text = json.loads(response['outputs'][0]['data'][0])
        # output can be a dict or a list of dict
        if isinstance(text, list):
            text = text[0]
        if self.client.task == "text-generation":
            text = text['generated_text']
            if text.startswith(prompt):
                text = text[len(prompt):]
        elif self.client.task == "text2text-generation":
            text = text['generated_text']
        elif self.client.task == "summarization":
            text = text['summary_text']
        elif self.client.task == "question-answering":
            text = text['answer']
        else:
            raise ValueError(
                f"Got invalid task {self.client.task}, "
                f"currently only {VALID_TASKS} are supported"
            )
        if stop is not None:
            text = enforce_stop_tokens(text, stop)
        return text

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings = []
        for text in texts:
            embeddings.append(self.embed_query(text))
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        response = self.client(inputs=text)
        if "error" in response:
            raise ValueError(
                f"Error raised by inference API: {response['error']}"
            )
        embeddings = response['outputs'][0]['data']

        return embeddings

if __name__=="__main__":
    endpoint_url = (
        "https://seldon-mesh.genai.sc.eng.hitachivantara.com"
    )
    llm = SeldonCore(
        repo_id="llama2-7b-chat-gpu",
        endpoint_url=endpoint_url,
        model_kwargs={
            "temperature": 0.1,
            "max_length": 128,
            "top_p": 0.15,
            "top_k": 0,
            "repetition_penalty": 1.1,
        }
    )
