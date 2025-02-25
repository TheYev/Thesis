from fastapi import FastAPI
from .routers import analyz


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

app.include_router(analyz.router)
