from pydantic_settings import BaseSettings
from typing import List


class kaiwaBaseSettings(BaseSettings):
    use_testnet: bool = False
    call_timeout: int = 60

    # TODO: whitelist&blacklist
    # whitelist: List[str] = []
    # blacklist: List[str] = []

    # class Config:
    #     env_prefix = "kaiwa"
    #     env_file = "env/config.env"
    #     extra = "ignore"
