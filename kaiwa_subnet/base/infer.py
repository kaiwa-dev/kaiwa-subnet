
from io import BytesIO
from typing import Optional
import base64
import threading
import vllm
import torch

from communex.module.module import Module, endpoint

class InferenceEngine(Module):
    def __init__(self, model_name: str = "stabilityai/sdxl-turbo") -> None:
        super().__init__()
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")

        self._lock = threading.Lock()

    @endpoint
    def sample(
        self, prompt: str, steps: int = 50, negative_prompt: str = "", seed:
    Optional[int]=None) -> str:
        generator = torch.Generator(self.device)
        if seed is None:
            seed = generator.seed()
        generator = generator.manual_seed(seed)
        with self._lock:
            image = self.pipeline(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                generator=generator,
                guidance_scale=0.0
            ).images[0]
        buf = BytesIO()
        image.save(buf, format="png")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode()

    @endpoint
    def get_metadata(self) -> dict:
        return {"model": self.model_name}

if __name__ == "__main__":
    d = InferenceEngine()
    out = d.sample(prompt="cat, jumping")
    with open("a.png", "wb") as f:
        f.write(out)
