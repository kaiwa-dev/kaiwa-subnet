import subprocess
import shlex
import httpx
import time

from communex.module.module import Module, endpoint
from loguru import logger


class InferenceEngine(Module):
    def __init__(self, models: list[str] = ["llama3"]) -> None:
        super().__init__()
        self.models = models
        self.endpoint = "http://127.0.0.1:11434"
        self.p = subprocess.Popen(shlex.split("ollama serve"))
        time.sleep(10)
        self.load_models()

    @endpoint
    def chat(self, input: dict, timeout: int = 120) -> dict:
        input["stream"] = False
        with httpx.Client() as client:
            resp = client.post(
                f"{self.endpoint}/api/chat",
                json=input,
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()

    def load_models(self):
        for model_name in self.models:
            try:
                logger.info(f"loading model {model_name}")
                with httpx.stream(
                    "POST",
                    f"{self.endpoint}/api/pull",
                    json={
                        "model": model_name,
                        "stream": True,
                    },
                    timeout=None,
                ) as resp:
                    for line in resp.iter_lines():
                        logger.info(f"ollama: {line}")
                    resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                logger.error(e)
                raise

    @endpoint
    def get_metadata(self) -> dict:
        return {"models": self.models}


if __name__ == "__main__":
    d = InferenceEngine()
    out = d.chat(
        {
            "model": "llama3",
            "messages": [{"role": "user", "content": "why is the sky blue?"}],
        }
    )
    print(out)
