from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class GenerateRequest(BaseModel):
    user_id: str
    message: str

@app.get("/")
def health():
    return {"status": "ai-service running"}

@app.post("/generate")
def generate(req: GenerateRequest):
    return {
        "reply": f"AI response for: {req.message}",
        "user_id": req.user_id
    }