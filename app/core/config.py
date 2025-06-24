from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import validator

class Settings(BaseSettings):
    # 基本设置
    ENV: str = "development"
    DEBUG: bool = True
    SECRET_KEY: str
    PROJECT_NAME: str = "FastAPI Project"
    API_V1_STR: str = "/api/v1"

    # 数据库设置
    DATABASE_URL: str

    class Config:
        env_file = ".env"

settings = Settings() 