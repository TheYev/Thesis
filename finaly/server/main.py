from fastapi import FastAPI
from .routers import analysis_video, auth
from .database import Base, engine


app = FastAPI()

Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(analysis_video.router)
app.include_router(auth.router)