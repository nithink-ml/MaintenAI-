"""
MaintenAI— API Gateway Authentication & Role-Based Access Control (RBAC)

Provides standard cryptographic HS256 JWT bearer token generation, verification,
pre-configured industrial role profiles (technician, engineer, manager, admin),
and FastAPI security dependencies.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import secrets
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from factorypulse.config import get_settings
from factorypulse.schemas import TokenResponse, UserResponse, UserRole

logger = logging.getLogger(__name__)

# Bearer scheme for OpenAPI docs and dependency injection
security = HTTPBearer(auto_error=False)


class LoginRequest(BaseModel):
    """User credentials schema for authentication."""

    username: str
    password: str


# PBKDF2-HMAC-SHA256 password hashing (stdlib-only, salted per user)
_PBKDF2_ITERATIONS = 390_000


def hash_password(password: str, *, salt: bytes | None = None, iterations: int | None = None) -> str:
    """Hash a password with PBKDF2-HMAC-SHA256. Returns 'pbkdf2_sha256$<iters>$<salt>$<hash>'."""
    salt = salt or secrets.token_bytes(16)
    iterations = iterations or _PBKDF2_ITERATIONS
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    """Verify a password against a stored PBKDF2 hash (or legacy unsalted sha256)."""
    if stored.startswith("pbkdf2_sha256$"):
        try:
            _, iterations_s, salt_hex, digest_hex = stored.split("$", 3)
            digest = hashlib.pbkdf2_hmac(
                "sha256", password.encode(), bytes.fromhex(salt_hex), int(iterations_s)
            )
            return secrets.compare_digest(digest.hex(), digest_hex)
        except (ValueError, TypeError):
            return False
    # Legacy development hash format (unsalted SHA-256); kept for pre-seeded users only
    return secrets.compare_digest(hashlib.sha256(password.encode()).hexdigest(), stored)


# Pre-seeded development users (development/demo credentials only — not a production user store)
_DEV_PASSWORD = "FactoryPulse2026!"


def _dev_user(
    user_id: str, username: str, role: UserRole, full_name: str
) -> dict[str, Any]:
    return {
        "id": user_id,
        "username": username,
        "email": f"{username}@factorypulse.internal",
        "password_hash": hash_password(_DEV_PASSWORD),
        "role": role,
        "full_name": full_name,
        "is_active": True,
        "created_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
    }


_USERS_DB: dict[str, dict[str, Any]] = {
    "technician": _dev_user(
        "USR-TECH-001", "technician", UserRole.TECHNICIAN, "Field Maintenance Technician"
    ),
    "engineer": _dev_user(
        "USR-ENG-002", "engineer", UserRole.ENGINEER, "Lead Reliability Engineer"
    ),
    "manager": _dev_user(
        "USR-MGR-003", "manager", UserRole.MANAGER, "Plant Operations Manager"
    ),
    "admin": _dev_user("USR-ADM-004", "admin", UserRole.ADMIN, "SCADA System Administrator"),
}


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def _b64decode(s: str) -> bytes:
    padding = 4 - (len(s) % 4)
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s.encode())


# Known-insecure default secret shipped in config; must never sign real tokens.
_INSECURE_DEFAULT_SECRETS = frozenset(
    {
        "CHANGE_ME_jwt_secret_key_at_least_32_chars",
        "factorypulse-industrial-scada-secret-key-2026",
    }
)

# Process-local secret generated when no secure secret is configured (development only).
_RUNTIME_JWT_SECRET: str | None = None


def get_jwt_secret() -> str:
    """
    Resolve the JWT signing secret from application settings (JWT_SECRET_KEY).

    - If a secure, non-default secret is configured, it is used.
    - In development, a random process-local secret is generated so tokens are
      still cryptographically signed (they will not survive a restart).
    - In production/staging, a missing or default secret is a hard failure.
    """
    global _RUNTIME_JWT_SECRET
    settings = get_settings()
    secret = settings.jwt.secret_key

    if secret and secret not in _INSECURE_DEFAULT_SECRETS:
        return secret

    if settings.app_env == "production":
        raise RuntimeError(
            "JWT_SECRET_KEY is unset or still the insecure default. "
            "Refusing to sign tokens in production. Set JWT_SECRET_KEY in the environment."
        )

    if _RUNTIME_JWT_SECRET is None:
        _RUNTIME_JWT_SECRET = secrets.token_urlsafe(48)
        logger.warning(
            "JWT_SECRET_KEY is unset or insecure; generated a random process-local secret. "
            "Issued tokens are valid for this process only. Set JWT_SECRET_KEY for persistence."
        )
    return _RUNTIME_JWT_SECRET


def create_access_token(
    user_id: str,
    username: str,
    role: UserRole,
    expires_delta: timedelta | None = None,
) -> TokenResponse:
    """Generate standard HS256 JWT bearer token."""
    secret = get_jwt_secret()
    settings = get_settings()
    lifetime = expires_delta or timedelta(minutes=settings.jwt.access_token_expire_minutes)
    exp = datetime.now(UTC) + lifetime
    exp_timestamp = int(exp.timestamp())

    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": user_id,
        "username": username,
        "role": str(role),
        "exp": exp_timestamp,
        "iat": int(datetime.now(UTC).timestamp()),
    }

    header_b64 = _b64encode(json.dumps(header, separators=(",", ":")).encode())
    payload_b64 = _b64encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()

    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = _b64encode(signature)

    token_str = f"{header_b64}.{payload_b64}.{sig_b64}"
    return TokenResponse(
        access_token=token_str,
        refresh_token=secrets.token_urlsafe(32),
        token_type="bearer",
        expires_in=int(lifetime.total_seconds()),
    )


def decode_access_token(token: str) -> dict[str, Any]:
    """Validate and decode HS256 JWT bearer token."""
    secret = get_jwt_secret()
    parts = token.split(".")
    if len(parts) != 3:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed bearer token format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    header_b64, payload_b64, sig_b64 = parts
    signing_input = f"{header_b64}.{payload_b64}".encode()
    expected_sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()

    # Enforce expected signing algorithm to prevent algorithm-confusion attacks
    try:
        header = json.loads(_b64decode(header_b64).decode())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Malformed bearer token header",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e
    if header.get("alg") != "HS256":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unsupported token algorithm",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        actual_sig = _b64decode(sig_b64)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature encoding",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token signature",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload_bytes = _b64decode(payload_b64)
        payload = json.loads(payload_bytes.decode())
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not decode token claims",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    # Check expiration
    exp = payload.get("exp")
    if exp and datetime.now(UTC).timestamp() > exp:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def authenticate_user(req: LoginRequest) -> UserResponse:
    """Validate username and password against user directory."""
    user_record = _USERS_DB.get(req.username.strip().lower())
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not verify_password(req.password, user_record["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user_record["is_active"]:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is deactivated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserResponse(
        id=user_record["id"],
        username=user_record["username"],
        email=user_record["email"],
        role=user_record["role"],
        full_name=user_record["full_name"],
        is_active=user_record["is_active"],
        created_at=user_record["created_at"],
    )


def get_current_user(
    auth: HTTPAuthorizationCredentials | None = Depends(security),
) -> UserResponse:
    """FastAPI dependency to extract and authenticate current user from Bearer header."""
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required: missing Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(auth.credentials)
    username = payload.get("username")
    user_record = _USERS_DB.get(username)
    if not user_record:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User referenced in token no longer exists",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return UserResponse(
        id=user_record["id"],
        username=user_record["username"],
        email=user_record["email"],
        role=user_record["role"],
        full_name=user_record["full_name"],
        is_active=user_record["is_active"],
        created_at=user_record["created_at"],
    )


def require_role(allowed_roles: list[UserRole]):
    """Decorator / dependency factory for endpoint RBAC."""

    def role_checker(user: UserResponse = Depends(get_current_user)) -> UserResponse:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Forbidden: Action requires one of roles: {[r.value for r in allowed_roles]}",
            )
        return user

    return role_checker
