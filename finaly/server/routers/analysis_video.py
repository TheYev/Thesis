from fastapi import APIRouter, Depends, HTTPException, Path
from typing import Annotated
from sqlalchemy.orm import Session
from starlette import status
from ..database import SessionLocal
from .auth import get_current_user
from ..models.videos import Videos
from ..models.users import Users
from finaly.model.train import video_test
from pydantic import BaseModel, Field


class videoRequest(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    input_path: str
    
class videoUpdateRequest(BaseModel):
    id: int
    name: str = Field(min_length=1, max_length=100)

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

db_dependency = Annotated[Session, Depends(get_db)]
user_dependency = Annotated[str, Depends(get_current_user)]

@router.post("/analysis", status_code=status.HTTP_200_OK)
async def analysis_video(videoRequest: videoRequest, user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
        
    #"C:/Users/thedi/OneDrive/Desktop/d/finaly/model/train/test_video.mp4"
    out_path = await video_test.process_video_with_tracking(videoRequest.input_path, show_video=False, save_video=True)
        
    create_video_model = Videos(
        input_path=videoRequest.input_path,
        name=videoRequest.name,
        output_path=out_path,
        user_id=user.get('id')
    )
    db.add(create_video_model)
    db.commit()
        
    return {"message": f"Video was analyzed, you can find this video on this path and video id: {out_path, create_video_model.id}"}
    
@router.get("/getUserVideos", status_code=status.HTTP_200_OK)
async def get_user_videos(user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    user_video = db.query(Users).filter(Users.id == user.get('id')).first()
    if user_video is None:
        raise HTTPException(status_code=404, detail="User not found")
    
    return_data = []
    for video in user_video.videos:
        return_data.append({"id": video.id, "name": video.name})
            
    return return_data

@router.get("/getVideo/{video_id}", status_code=status.HTTP_200_OK)
async def get_video_by_id(user: user_dependency, db: db_dependency, video_id: int = Path(gt=0)):
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    video = db.query(Videos).filter(Videos.id == video_id)\
        .filter(Videos.user_id == user.get('id')).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    return {"id": video.id, "name": video.name, "input_path": video.input_path, "output_path": video.output_path}
    
@router.delete("/deleteVideo/{video_id}", status_code=status.HTTP_200_OK)
async def delete_video(user: user_dependency, db: db_dependency, video_id: int = Path(gt=0)):
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    video = db.query(Videos).filter(Videos.id == video_id).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    db.query(Videos).filter(Videos.id == video_id)\
        .filter(Videos.user_id == user.get('id'))\
        .delete()
    db.commit()
    return {"message": f"Video was deleted, id: {video_id}"}

@router.put("/updateVideo", status_code=status.HTTP_200_OK)
async def update_video(videoUpdateRequest: videoUpdateRequest, user: user_dependency, db: db_dependency):
    if user is None:
        raise HTTPException(status_code=401, detail="Unauthorized")
    
    video = db.query(Videos).filter(Videos.id == videoUpdateRequest.id)\
        .filter(Videos.user_id == user.get('id')).first()
    if video is None:
        raise HTTPException(status_code=404, detail="Video not found")
    
    video.name = videoUpdateRequest.name
    db.add(video)
    db.commit()
    return {"message": f"Video was updated, id: {videoUpdateRequest.id}"}
   