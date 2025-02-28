from ..database import Base
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship

class Users(Base):
    __tablename__ = 'users'
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True)
    username = Column(String, unique=True)
    hashed_password = Column(String)
    
    videos = relationship('Videos', back_populates='user')