from typing import cast
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from pydantic import BaseModel
from core.database import get_db
from core.security import verify_password, create_access_token
from models.user import User, UserRole

router = APIRouter()


class Token(BaseModel):
    access_token: str
    token_type: str
    user_id: int
    role: str
    full_name: str
    email: str


def _normalize_role(role: UserRole | str) -> str:
    value = role.value if isinstance(role, UserRole) else str(role)
    return "admin" if value == "admin" else "user"


@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(User).where(User.email == form_data.username)
    )
    user = result.scalar_one_or_none()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not verify_password(form_data.password, cast(str, user.hashed_password)):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
        )

    if not cast(bool, user.is_active):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Account is disabled",
        )

    token = create_access_token({"sub": str(cast(int, user.id))})

    return Token(
        access_token=token,
        token_type="bearer",
        user_id=cast(int, user.id),
        role=_normalize_role(cast(UserRole, user.role)),
        full_name=cast(str, user.full_name),
        email=cast(str, user.email),
    )


