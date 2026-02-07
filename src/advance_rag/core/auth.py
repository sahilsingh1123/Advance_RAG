"""Authentication and authorization utilities."""

import jwt
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from passlib.context import CryptContext
from advance_rag.core.config import get_settings
from advance_rag.core.exceptions import AuthenticationError, AuthorizationError

settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class TokenManager:
    """JWT token management."""

    def __init__(self):
        self.secret_key = settings.SECRET_KEY
        self.algorithm = settings.JWT_ALGORITHM
        self.expire_minutes = settings.JWT_EXPIRE_MINUTES

    def create_access_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=self.expire_minutes)

        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt

    def verify_token(self, token: str) -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            raise AuthenticationError("Token has expired")
        except jwt.JWTError:
            raise AuthenticationError("Invalid token")

    def create_refresh_token(
        self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()

        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(days=7)  # Refresh tokens last longer

        to_encode.update({"exp": expire, "type": "refresh"})
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt


class PasswordManager:
    """Password hashing and verification."""

    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password using bcrypt."""
        return pwd_context.hash(password)

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, hashed_password)


class User:
    """User model for authentication."""

    def __init__(
        self,
        id: str,
        email: str,
        name: str,
        role: str,
        studies: List[str],
        permissions: Optional[List[str]] = None,
    ):
        self.id = id
        self.email = email
        self.name = name
        self.role = role
        self.studies = studies
        self.permissions = permissions or []
        self.is_active = True
        self.last_login = None

    def has_permission(self, permission: str) -> bool:
        """Check if user has specific permission."""
        return permission in self.permissions

    def has_study_access(self, study_id: str) -> bool:
        """Check if user has access to specific study."""
        return study_id in self.studies or "admin" in self.permissions

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary."""
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "role": self.role,
            "studies": self.studies,
            "permissions": self.permissions,
            "is_active": self.is_active,
            "last_login": self.last_login.isoformat() if self.last_login else None,
        }


class Role:
    """Role-based access control."""

    ROLES = {
        "admin": {
            "permissions": [
                "read",
                "write",
                "delete",
                "ingest",
                "manage_users",
                "manage_studies",
                "view_all_studies",
                "system_admin",
            ]
        },
        "data_manager": {"permissions": ["read", "write", "ingest", "manage_studies"]},
        "analyst": {"permissions": ["read", "query", "export"]},
        "viewer": {"permissions": ["read"]},
    }

    @classmethod
    def get_permissions(cls, role: str) -> List[str]:
        """Get permissions for role."""
        return cls.ROLES.get(role, {}).get("permissions", [])

    @classmethod
    def has_permission(cls, role: str, permission: str) -> bool:
        """Check if role has permission."""
        return permission in cls.get_permissions(role)


class AuthService:
    """Authentication service."""

    def __init__(self):
        self.token_manager = TokenManager()
        self.password_manager = PasswordManager()
        # In production, use a proper user store
        self.users: Dict[str, User] = {}

    def register_user(
        self,
        email: str,
        password: str,
        name: str,
        role: str = "viewer",
        studies: Optional[List[str]] = None,
    ) -> User:
        """Register a new user."""
        # Check if user already exists
        if any(user.email == email for user in self.users.values()):
            raise AuthenticationError("User already exists")

        # Create user
        user_id = f"user_{len(self.users) + 1}"
        hashed_password = self.password_manager.hash_password(password)

        user = User(
            id=user_id,
            email=email,
            name=name,
            role=role,
            studies=studies or [],
            permissions=Role.get_permissions(role),
        )

        self.users[user_id] = user
        return user

    def authenticate(self, email: str, password: str) -> User:
        """Authenticate user with email and password."""
        user = None
        for u in self.users.values():
            if u.email == email:
                user = u
                break

        if not user or not user.is_active:
            raise AuthenticationError("Invalid credentials")

        # In production, verify against stored hash
        # For now, we'll skip password verification for demo

        user.last_login = datetime.utcnow()
        return user

    def create_tokens(self, user: User) -> Dict[str, str]:
        """Create access and refresh tokens for user."""
        access_token_data = {
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "type": "access",
        }

        refresh_token_data = {"sub": user.id, "type": "refresh"}

        access_token = self.token_manager.create_access_token(access_token_data)
        refresh_token = self.token_manager.create_refresh_token(refresh_token_data)

        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
        }

    def refresh_access_token(self, refresh_token: str) -> str:
        """Create new access token from refresh token."""
        payload = self.token_manager.verify_token(refresh_token)

        if payload.get("type") != "refresh":
            raise AuthenticationError("Invalid refresh token")

        user_id = payload.get("sub")
        user = self.users.get(user_id)

        if not user or not user.is_active:
            raise AuthenticationError("User not found")

        access_token_data = {
            "sub": user.id,
            "email": user.email,
            "role": user.role,
            "type": "access",
        }

        return self.token_manager.create_access_token(access_token_data)

    def get_current_user(self, token: str) -> User:
        """Get current user from token."""
        payload = self.token_manager.verify_token(token)

        if payload.get("type") != "access":
            raise AuthenticationError("Invalid token type")

        user_id = payload.get("sub")
        user = self.users.get(user_id)

        if not user or not user.is_active:
            raise AuthenticationError("User not found")

        return user


# Global auth service instance
auth_service = AuthService()


def require_permission(permission: str):
    """Decorator to require specific permission."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Get user from request (would be set by middleware)
            user = kwargs.get("current_user")
            if not user:
                raise AuthorizationError("Authentication required")

            if not user.has_permission(permission):
                raise AuthorizationError(f"Permission '{permission}' required")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


def require_study_access(study_id_param: str = "study_id"):
    """Decorator to require study access."""

    def decorator(func):
        async def wrapper(*args, **kwargs):
            user = kwargs.get("current_user")
            if not user:
                raise AuthorizationError("Authentication required")

            study_id = kwargs.get(study_id_param)
            if study_id and not user.has_study_access(study_id):
                raise AuthorizationError(f"Access to study '{study_id}' required")

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# Create default admin user
def create_default_admin():
    """Create default admin user."""
    try:
        auth_service.register_user(
            email="admin@example.com",
            password="admin123",
            name="System Administrator",
            role="admin",
            studies=[],
        )
    except AuthenticationError:
        pass  # User already exists
