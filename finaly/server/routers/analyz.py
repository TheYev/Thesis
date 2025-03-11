from fastapi import APIRouter, Depends, HTTPException
from typing import Annotated
from sqlalchemy.orm import Session
from ..database import SessionLocal
from finaly.model.train import video_test
from starlette import status

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

@router.post("/", status_code=status.HTTP_200_OK)
async def analyz(path: str, db: db_dependency):
    try:
        #"C:/Users/thedi/OneDrive/Desktop/d/finaly/model/train/test_video.mp4"
        out_path = await video_test.process_video_with_tracking(path, show_video=False, save_video=True)
        return {"message": f"body: {out_path}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
   