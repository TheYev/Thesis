from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from pydantic import BaseModel, Field
from passlib.context import CryptContext
from starlette import status
from typing import Annotated
from sqlalchemy.orm import Session
from jose import jwt, JWTError
from datetime import timedelta, datetime, timezone
from ..database import SessionLocal
from ..models.users import Users 


router = APIRouter(
    prefix="/auth",
    tags=["auth"],
)

SECRET_KEY = 'add-secret-key-from-env'
ALGORITM = 'HS256'

bcrypt_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_beare = OAuth2PasswordBearer(tokenUrl="auth/token")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        
db_dependency = Annotated[Session, Depends(get_db)]

class CreateUserRequest(BaseModel):
    email: str = Field(min_length=1, max_length=100)
    username: str = Field(min_length=1, max_length=20)
    password: str = Field(min_length=1, max_length=50)
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
def authenicate_user(username: str, password: str, db: db_dependency):
    user = db.query(Users).filter(Users.username == username).first()
    if not user:
        return False
    if not bcrypt_context.verify(password, user.hashed_password):
        return False
    return user

def create_access_token(username: str, user_id: int, expires_delta: timedelta):
    encode = {'sub': username, 'id': user_id}
    expires = datetime.now(timezone.utc) + expires_delta
    encode.update({'exp': expires})
    return jwt.encode(encode, SECRET_KEY, algorithm=ALGORITM)

def get_current_user(token: Annotated[str, Depends(oauth2_beare)]):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITM])
        username: str = payload.get('sub')
        user_id: int = payload.get('id')
        if username is None or user_id is None:
            raise HTTPException(status_code=401,
                                detail="Could not validate credentials")
        return {"username": username, "id": user_id}
    except JWTError:
        raise HTTPException(status_code=401,
                            detail="Could not validate credentials")

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_user(user: CreateUserRequest, db: db_dependency):
    create_user_model = Users(
        email=user.email,
        username=user.username,
        hashed_password=bcrypt_context.hash(user.password)
    )
    
    db.add(create_user_model)
    db.commit()
    return {"User": create_user_model.username}
    
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
                                 db: db_dependency):
    user = authenicate_user(form_data.username, form_data.password, db)
    if not user:
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    
    token = create_access_token(user.username, user.id, timedelta(minutes=15))
    
    return {"access_token": token, "token_type": "bearer"}
