from ..database import Base
from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship

class Videos(Base):
    __tablename__ = 'videos'
    
    id = Column(Integer, primary_key=True, index=True)
    input_path = Column(String, nullable=False)
    output_path = Column(String, nullable=False)
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))

    owner = relationship("Users", back_populates="videos")