from kaiwa_subnet.base.config import kaiwaBaseSettings
from typing import List


class MinerSettings(kaiwaBaseSettings):
    host: str
    port: int
    model: str = "stabilityai/sdxl-turbo"
