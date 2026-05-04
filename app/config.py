from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    db_server: str = "163.178.173.130"
    db_port: int = 1433
    db_name: str = "MovieRecommenderDB"
    db_user: str = "lenguajes"
    db_password: str = "lenguajesparaiso2025"

    # App
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_env: str = "development"

    # Modelo
    model_path: str = "data/svd_model.pkl"
    top_n_recommendations: int = 10

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()