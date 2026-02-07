"""Authentication endpoints."""

from datetime import timedelta
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

from advance_rag.core.auth import auth_service, User, create_default_admin
from advance_rag.core.middleware import get_current_user, get_admin_user
from advance_rag.core.exceptions import AuthenticationError

router = APIRouter()


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int


class UserCreate(BaseModel):
    """User creation model."""

    email: EmailStr
    password: str
    name: str
    role: str = "viewer"
    studies: list[str] = []


class UserResponse(BaseModel):
    """User response model."""

    id: str
    email: str
    name: str
    role: str
    studies: list[str]
    permissions: list[str]
    is_active: bool
    last_login: str | None = None


class LoginRequest(BaseModel):
    """Login request model."""

    email: EmailStr
    password: str


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str


@router.post("/register", response_model=UserResponse)
async def register(user_data: UserCreate):
    """Register a new user."""
    try:
        user = auth_service.register_user(
            email=user_data.email,
            password=user_data.password,
            name=user_data.name,
            role=user_data.role,
            studies=user_data.studies,
        )
        return UserResponse(**user.to_dict())
    except AuthenticationError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/login", response_model=TokenResponse)
async def login(login_data: LoginRequest):
    """Login user and return tokens."""
    try:
        user = auth_service.authenticate(login_data.email, login_data.password)
        tokens = auth_service.create_tokens(user)

        # Set expires_in (in seconds)
        expires_in = auth_service.token_manager.expire_minutes * 60

        return TokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            expires_in=expires_in,
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.post("/refresh", response_model=Dict[str, str])
async def refresh_token(refresh_data: RefreshTokenRequest):
    """Refresh access token."""
    try:
        access_token = auth_service.refresh_access_token(refresh_data.refresh_token)
        return {"access_token": access_token}
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user),
):
    """Get current user information."""
    return UserResponse(**current_user.to_dict())


@router.get("/users", response_model=list[UserResponse])
async def list_users(current_user: User = Depends(get_admin_user)):
    """List all users (admin only)."""
    users = []
    for user in auth_service.users.values():
        users.append(UserResponse(**user.to_dict()))
    return users


@router.post("/init-admin")
async def initialize_admin():
    """Initialize default admin user."""
    create_default_admin()
    return {"message": "Admin user initialized"}
