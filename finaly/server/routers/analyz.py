from fastapi import APIRouter, Depends
from typing import Annotated
from sqlalchemy.orm import Session
from ..database import SessionLocal
from finaly.model.train import video_test

router = APIRouter(
    prefix="/analyze",
    tags=["analyze"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]

@router.post("/")
async def analyz():
    await video_test.process_video_with_tracking("C:/Users/thedi/OneDrive/Desktop/d/finaly/model/train/test_video.mp4", show_video=False, save_video=True)
    return {"message": "Analyzing video"}