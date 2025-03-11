from fastapi import FastAPI
from .routers import analyz, auth
from .database import Base, engine


app = FastAPI()

Base.metadata.create_all(bind=engine)

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(analyz.router)
app.include_router(auth.router)