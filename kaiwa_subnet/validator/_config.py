from kaiwa_subnet.base.config import kaiwaBaseSettings
from typing import List


class ValidatorSettings(kaiwaBaseSettings):
    host: str = "0.0.0.0"
    port: int = 0
    iteration_interval: int = 60
