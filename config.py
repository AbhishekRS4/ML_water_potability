from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Water Potability API"
    version: str = "2024.02.07"


settings = Settings()
