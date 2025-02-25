from fastapi import APIRouter
from finaly.model.train import video_test


router = APIRouter(
    prefix="/analyze",
    tags=["analyze"],
)

@router.post("/")
async def analyz():
    video_test.process_video_with_tracking("C:/Users/thedi/OneDrive/Desktop/d/finaly/model/train/test_video.mp4", show_video=False, save_video=True)
    return {"message": "Analyzing video"}