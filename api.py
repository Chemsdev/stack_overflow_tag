from fastapi import FastAPI




app = FastAPI()

@app.post("/data")
async def first():
    print("hello world")
    return 


