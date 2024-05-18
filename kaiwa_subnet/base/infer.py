
import threading
from transformers import pipeline
import torch

from communex.module.module import Module, endpoint

class InferenceEngine(Module):
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.pipeline = pipeline("text-generation", model_name, torch_dtype=torch.bfloat16, device_map="auto")
        self._lock = threading.Lock()

    @endpoint
    def chat(
        self, chat: list, generate_kwargs: dict=None) -> str:
        if not generate_kwargs:
            generate_kwargs = dict(
                temperature = 0.7,
                max_new_tokens = 512
            )
        with self._lock:
            response = self.pipeline(chat,  generate_kwargs=generate_kwargs)
        return response[0]['generated_text'][-1]['content']

    @endpoint
    def get_metadata(self) -> dict:
        return {"model": self.model_name}

if __name__ == "__main__":
    d = InferenceEngine()
    out = d.chat([
        {"role": "system", "content": "You are a sassy, wise-cracking robot as imagined by Hollywood circa 1986."},
        {"role": "user", "content": "Hey, can you tell me any fun things to do in New York?"}
    ])
    print(out)
