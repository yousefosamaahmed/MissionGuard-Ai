from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Local .env values fill missing settings, while real server/container
# environment variables always take priority.
load_dotenv(PROJECT_ROOT / ".env", override=False)

_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_TRUE_VALUES = {"1", "true", "yes", "on", "enabled"}
_FALSE_VALUES = {"0", "false", "no", "off", "disabled"}


def _env_flag(name: str, default: bool = False) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None or not raw_value.strip():
        return default

    clean_value = raw_value.strip().lower()

    if clean_value in _TRUE_VALUES:
        return True

    if clean_value in _FALSE_VALUES:
        return False

    raise RuntimeError(
        f"{name} must be one of: "
        "true/false, yes/no, on/off, 1/0."
    )


def database_requested() -> bool:
    """
    Return whether PostgreSQL support was explicitly enabled.

    Local MissionGuard analysis works without PostgreSQL. The database is
    enabled when DATABASE_ENABLED is true, or when legacy database variables
    are already provided and DATABASE_ENABLED is not explicitly false.
    """

    explicit_value = os.getenv("DATABASE_ENABLED")

    if explicit_value is not None and explicit_value.strip():
        return _env_flag("DATABASE_ENABLED", default=False)

    return bool(
        os.getenv("DATABASE_URL", "").strip()
        or os.getenv("POSTGRES_USER", "").strip()
    )


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw_value = os.getenv(name, str(default)).strip()

    try:
        value = int(raw_value)
    except ValueError as error:
        raise RuntimeError(f"{name} must be a valid integer.") from error

    if value < minimum:
        raise RuntimeError(f"{name} must be at least {minimum}.")

    return value


def _validated_schema(value: str) -> str:
    clean_value = value.strip()

    if not _IDENTIFIER_PATTERN.fullmatch(clean_value):
        raise RuntimeError(
            "POSTGRES_SCHEMA must be a simple PostgreSQL identifier "
            f"(letters, digits and underscores). Received: {value!r}"
        )

    return clean_value


def _normalise_database_url(raw_url: str) -> URL:
    clean_url = raw_url.strip()

    if clean_url.startswith("postgres://"):
        clean_url = "postgresql://" + clean_url[len("postgres://") :]

    url = make_url(clean_url)

    if not url.drivername.startswith("postgresql"):
        raise RuntimeError(
            "DATABASE_URL must point to PostgreSQL. "
            f"Received driver: {url.drivername!r}"
        )

    # The project installs psycopg 3, so normalise provider URLs that omit
    # a SQLAlchemy driver suffix.
    if url.drivername in {
        "postgresql",
        "postgresql+psycopg2",
        "postgresql+pg8000",
    }:
        url = url.set(drivername="postgresql+psycopg")

    if url.drivername != "postgresql+psycopg":
        raise RuntimeError(
            "MissionGuard requires the SQLAlchemy psycopg driver. "
            f"Received: {url.drivername!r}"
        )

    return url


@dataclass(frozen=True)
class DatabaseSettings:
    url: URL
    schema: str
    pool_size: int
    max_overflow: int
    pool_timeout: int
    pool_recycle: int
    maintenance_database: str

    @property
    def user(self) -> str | None:
        return self.url.username

    @property
    def password(self) -> str | None:
        return self.url.password

    @property
    def host(self) -> str | None:
        return self.url.host

    @property
    def port(self) -> int | None:
        return self.url.port

    @property
    def database(self) -> str | None:
        return self.url.database

    @property
    def sslmode(self) -> str | None:
        value = self.url.query.get("sslmode")
        return str(value) if value is not None else None

    def sqlalchemy_connect_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "options": f"-c search_path={self.schema},public",
        }

        connect_timeout = os.getenv(
            "POSTGRES_CONNECT_TIMEOUT",
            "5",
        ).strip()

        if connect_timeout:
            try:
                args["connect_timeout"] = max(
                    1,
                    int(connect_timeout),
                )
            except ValueError as error:
                raise RuntimeError(
                    "POSTGRES_CONNECT_TIMEOUT must be a valid integer."
                ) from error

        return args

    def psycopg_connect_kwargs(
        self,
        database_name: str | None = None,
        include_search_path: bool = False,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}

        if self.host:
            kwargs["host"] = self.host
        if self.port:
            kwargs["port"] = self.port
        if self.user:
            kwargs["user"] = self.user
        if self.password:
            kwargs["password"] = self.password

        target_database = database_name or self.database
        if target_database:
            kwargs["dbname"] = target_database

        # Preserve standard libpq-style query parameters from managed
        # PostgreSQL connection strings, including sslmode.
        for key, value in self.url.query.items():
            if key == "options":
                continue
            kwargs[str(key)] = str(value)

        if include_search_path:
            kwargs["options"] = f"-c search_path={self.schema},public"

        return kwargs


def get_database_settings() -> DatabaseSettings:
    if not database_requested():
        raise RuntimeError(
            "PostgreSQL is disabled. MissionGuard can run in local mode. "
            "Set DATABASE_ENABLED=true and configure DATABASE_URL or the "
            "POSTGRES_* variables to enable database persistence."
        )

    raw_database_url = os.getenv("DATABASE_URL", "").strip()

    if raw_database_url:
        url = _normalise_database_url(raw_database_url)
    else:
        user = os.getenv("POSTGRES_USER", "").strip()
        password = os.getenv("POSTGRES_PASSWORD", "").strip()
        host = os.getenv("POSTGRES_HOST", "127.0.0.1").strip()
        database = os.getenv("POSTGRES_DB", "missionguard_ai").strip()
        port = _env_int("POSTGRES_PORT", 5432, minimum=1)

        if not user:
            raise RuntimeError(
                "Set DATABASE_URL or POSTGRES_USER when "
                "DATABASE_ENABLED=true."
            )

        if not password:
            raise RuntimeError(
                "Set DATABASE_URL or POSTGRES_PASSWORD when "
                "DATABASE_ENABLED=true."
            )

        query: dict[str, str] = {}
        sslmode = os.getenv("POSTGRES_SSLMODE", "").strip()
        connect_timeout = os.getenv("POSTGRES_CONNECT_TIMEOUT", "").strip()

        if sslmode:
            query["sslmode"] = sslmode
        if connect_timeout:
            query["connect_timeout"] = connect_timeout

        url = URL.create(
            drivername="postgresql+psycopg",
            username=user,
            password=password,
            host=host,
            port=port,
            database=database,
            query=query,
        )

    if not url.database:
        raise RuntimeError("The PostgreSQL database name is missing.")

    schema = _validated_schema(
        os.getenv("POSTGRES_SCHEMA", "missionguard")
    )

    return DatabaseSettings(
        url=url,
        schema=schema,
        pool_size=_env_int("POSTGRES_POOL_SIZE", 5, minimum=1),
        max_overflow=_env_int("POSTGRES_MAX_OVERFLOW", 10, minimum=0),
        pool_timeout=_env_int("POSTGRES_POOL_TIMEOUT", 10, minimum=1),
        pool_recycle=_env_int("POSTGRES_POOL_RECYCLE", 1800, minimum=0),
        maintenance_database=os.getenv(
            "POSTGRES_MAINTENANCE_DB",
            "postgres",
        ).strip()
        or "postgres",
    )
