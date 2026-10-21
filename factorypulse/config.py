"""
MaintenAI— Centralized Application Settings

All configuration is loaded from environment variables with sensible defaults.
Uses pydantic-settings for typed, validated configuration management.
"""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _EnvFileBaseSettings(BaseSettings):
    """Base for sub-settings: also reads variables from the project .env file.

    Without an explicit env_file, nested BaseSettings models only read real
    environment variables — .env values would be silently ignored for every
    sub-configuration (JWT secrets, DB credentials, Redis, etc.).
    """

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")


class DatabaseSettings(_EnvFileBaseSettings):
    """TimescaleDB / PostgreSQL connection settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="POSTGRES_")

    host: str = "localhost"
    port: int = 5432
    db: str = "factorypulse"
    user: str = "factorypulse"
    password: str = "factorypulse_dev"

    @property
    def async_url(self) -> str:
        return f"postgresql+asyncpg://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"

    @property
    def sync_url(self) -> str:
        return (
            f"postgresql+psycopg2://{self.user}:{self.password}@{self.host}:{self.port}/{self.db}"
        )


class RedisSettings(_EnvFileBaseSettings):
    """Redis connection settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="REDIS_")

    host: str = "localhost"
    port: int = 6379
    password: str = "factorypulse_dev"
    db: int = 0

    @property
    def url(self) -> str:
        return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"


class QdrantSettings(_EnvFileBaseSettings):
    """Qdrant vector database settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="QDRANT_")

    host: str = "localhost"
    port: int = 6333
    api_key: str | None = None
    collection_name: str = "maintenance_docs"


class MLflowSettings(_EnvFileBaseSettings):
    """MLflow tracking and registry settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="MLFLOW_")

    tracking_uri: str = "http://localhost:5000"
    artifact_root: str = "./mlruns"


class JWTSettings(_EnvFileBaseSettings):
    """JWT authentication settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="JWT_")

    secret_key: str = "CHANGE_ME_jwt_secret_key_at_least_32_chars"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7


class LLMSettings(_EnvFileBaseSettings):
    """LLM provider settings."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="LLM_")

    provider: Literal["openai", "gemini", "ollama", "openrouter"] = "ollama"
    model: str = "llama3.1:8b"
    api_key: str = ""
    base_url: str = "http://localhost:11434/v1"


class AlertSettings(_EnvFileBaseSettings):
    """Deterministic alert engine thresholds."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", env_prefix="ALERT_")

    anomaly_threshold: float = 0.85
    rul_threshold: int = 15
    failure_probability_threshold: float = 0.70
    hysteresis_cycles: int = 3
    dedup_window_minutes: int = 15


class Settings(BaseSettings):
    """Root settings aggregating all sub-configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_name: str = "FactoryPulse"
    app_env: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    log_level: str = "INFO"

    # CORS
    cors_origins: str = "http://localhost:3000,http://localhost:8000"

    # Service ports
    api_gateway_port: int = 8000
    inference_service_port: int = 8001
    copilot_service_port: int = 8002
    ingestion_service_port: int = 8003

    # Sub-configs
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    qdrant: QdrantSettings = Field(default_factory=QdrantSettings)
    mlflow: MLflowSettings = Field(default_factory=MLflowSettings)
    jwt: JWTSettings = Field(default_factory=JWTSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    alerts: AlertSettings = Field(default_factory=AlertSettings)

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",")]


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
