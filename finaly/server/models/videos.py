from ..database import Base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

class Videos(Base):
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True, index=True)
    input_path = Column(String)
    output_path = Column(String)
    user_id = Column(Integer, ForeignKey('users.id'))
    
    user = relationship('Users', back_populates='videos')