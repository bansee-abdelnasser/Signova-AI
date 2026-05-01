from fastapi import APIRouter
from pydantic import BaseModel
from app.services.text_to_sign_service import generate_video

router = APIRouter()

class TextRequest(BaseModel):
    text: str


@router.post("/")
def text_to_sign(req: TextRequest):
    video_path = generate_video(req.text)
    return {"video_path": video_path}