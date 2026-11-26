
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Tanzania Foundation AI API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

@app.get("/")
def home():
    return {
        "name": "Tanzania Foundation AI",
        "status": "API is running"
    }

@app.get("/health")
def health():
    return {
        "api": "online",
        "model_ready": False,
        "message": "Model bado haijafundishwa."
    }

@app.post("/chat")
def chat(request: ChatRequest):
    message = request.message.strip()

    if not message:
        raise HTTPException(
            status_code=400,
            detail="Ujumbe hauwezi kuwa tupu."
        )

    raise HTTPException(
        status_code=503,
        detail="Model bado haijaunganishwa."
    )
