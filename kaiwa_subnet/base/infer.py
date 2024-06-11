import subprocess
import shlex
import httpx
import time
import asyncio

from communex.module.module import Module, endpoint
from loguru import logger

from starlette.responses import StreamingResponse, JSONResponse

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_engine import LoRAModulePath


class InferenceEngine(Module):
    def __init__(self, model: str = "NousResearch/Meta-Llama-3-8B-Instruct") -> None:
        super().__init__()
        engine_args = AsyncEngineArgs(
            model=model,
            enforce_eager=True,
            max_model_len=2048,
            quantization="fp8",
        )
        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        model_config = asyncio.run(self.engine.get_model_config())
        served_model_names = [engine_args.model]
        response_role = ""
        lora_modules = None
        chat_template = None
        self.openai_serving_chat = OpenAIServingChat(
            self.engine,
            model_config,
            served_model_names,
            response_role,
            lora_modules,
            chat_template,
        )

    @endpoint
    def chat(
        self, request: ChatCompletionRequest, timeout: int = 120
    ) -> ChatCompletionResponse:
        resp = asyncio.run(self.openai_serving_chat.create_chat_completion(request))
        return resp

    @endpoint
    def get_metadata(self) -> dict:
        return {"models": self.models}


if __name__ == "__main__":
    d = InferenceEngine()
    out = d.chat(
        request=ChatCompletionRequest(
            model="NousResearch/Meta-Llama-3-8B-Instruct",
            messages=[{"role": "user", "content": "Hello!"}],
        )
    )
    print(out)
