from datetime import datetime
from typing import Dict, List, cast
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from core.database import get_db
from core.security import hash_password
from core.security import decode_access_token
from models.user import User, UserRole

router = APIRouter()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")


class UserCreateRequest(BaseModel):
    email: EmailStr
    full_name: str
    role: Literal["admin", "user"] = "user"
    password: str = "changeme123"


class UserUpdateRequest(BaseModel):
    email: EmailStr
    full_name: str
    role: Literal["admin", "user"]


class ToggleActiveRequest(BaseModel):
    is_active: bool


class PasswordChangeRequest(BaseModel):
    new_password: str


PERMISSIONS: List[Dict[str, object]] = [
    {"feature": "View Dashboard", "admin": True, "user": True},
    {"feature": "View Sales Analytics", "admin": True, "user": True},
    {"feature": "View Inventory", "admin": True, "user": True},
    {"feature": "View Customers", "admin": True, "user": True},
    {"feature": "ML Forecasting", "admin": True, "user": True},
    {"feature": "AI Chatbot", "admin": True, "user": True},
    {"feature": "Export CSV / PDF", "admin": True, "user": True},
    {"feature": "Create Reports", "admin": True, "user": True},
    {"feature": "Manage Users", "admin": True, "user": False},
    {"feature": "System Settings", "admin": True, "user": False},
    {"feature": "View Audit Logs", "admin": True, "user": False},
    {"feature": "API Access", "admin": True, "user": True},
]


def _normalize_role(role: UserRole | str) -> str:
    value = role.value if isinstance(role, UserRole) else str(role)
    return "admin" if value == "admin" else "user"


def _to_storage_role(role: Literal["admin", "user"]) -> UserRole:
    return UserRole.admin if role == "admin" else UserRole.user




async def get_admin_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    user_id = decode_access_token(token)
    if not user_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

    result = await db.execute(select(User).where(User.id == int(user_id)))
    user = result.scalar_one_or_none()
    if not user or not cast(bool, user.is_active):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Inactive or missing user")

    if cast(UserRole, user.role) != UserRole.admin:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")

    return user


def _serialize_user(user: User) -> Dict[str, object]:
    created_at = cast(datetime | None, user.created_at)
    updated_at = cast(datetime | None, user.updated_at)
    created = created_at.date().isoformat() if created_at else None
    last_login = updated_at.isoformat() if updated_at else "Never"
    role_value = _normalize_role(cast(UserRole, user.role))
    return {
        "id": cast(int, user.id),
        "email": cast(str, user.email),
        "full_name": cast(str, user.full_name),
        "role": role_value,
        "is_active": cast(bool, user.is_active),
        "last_login": last_login,
        "created": created,
    }


@router.get("/users")
async def list_users(
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).order_by(User.id.asc()))
    users = result.scalars().all()
    return [_serialize_user(u) for u in users]


@router.post("/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreateRequest,
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = User(
        email=payload.email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=_to_storage_role(payload.role),
        is_active=True,
    )
    db.add(user)
    await db.flush()
    return _serialize_user(user)


@router.put("/users/{user_id}")
async def update_user(
    user_id: int,
    payload: UserUpdateRequest,
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    existing = await db.execute(select(User).where(User.email == payload.email, User.id != user_id))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=400, detail="Email already exists")

    setattr(user, "email", payload.email)
    setattr(user, "full_name", payload.full_name)
    setattr(user, "role", _to_storage_role(payload.role))
    setattr(user, "updated_at", datetime.utcnow())
    await db.flush()
    return _serialize_user(user)


@router.patch("/users/{user_id}/active")
async def set_user_active(
    user_id: int,
    payload: ToggleActiveRequest,
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    if cast(int, admin_user.id) == user_id and not payload.is_active:
        raise HTTPException(status_code=400, detail="You cannot deactivate your own account")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    setattr(user, "is_active", payload.is_active)
    setattr(user, "updated_at", datetime.utcnow())
    await db.flush()
    return _serialize_user(user)


@router.post("/users/{user_id}/password")
async def change_user_password(
    user_id: int,
    payload: PasswordChangeRequest,
    _: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    if len(payload.new_password) < 6:
        raise HTTPException(status_code=400, detail="Password must be at least 6 characters")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    setattr(user, "hashed_password", hash_password(payload.new_password))
    setattr(user, "updated_at", datetime.utcnow())
    await db.flush()
    return {"message": "Password changed successfully"}


@router.delete("/users/{user_id}")
async def delete_user(
    user_id: int,
    admin_user: User = Depends(get_admin_user),
    db: AsyncSession = Depends(get_db),
):
    if cast(int, admin_user.id) == user_id:
        raise HTTPException(status_code=400, detail="You cannot delete your own account")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    await db.delete(user)
    return {"message": "User deleted successfully"}


@router.get("/permissions")
async def get_permissions(_: User = Depends(get_admin_user)):
    return PERMISSIONS
