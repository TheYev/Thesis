from ..database import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from .videos import Videos

class Users(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)

    videos = relationship(Videos, back_populates="owner", cascade="all, delete-orphan")