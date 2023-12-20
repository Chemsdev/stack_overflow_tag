from fastapi import FastAPI




app = FastAPI()

@app.post("/data")
async def first():
    return "hello world"