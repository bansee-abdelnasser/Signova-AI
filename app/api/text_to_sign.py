from fastapi import APIRouter
from fastapi.responses import FileResponse
from pydantic import BaseModel
from app.services.text_to_sign_service import generate_video

router = APIRouter()

class TextRequest(BaseModel):
    text: str


@router.post("/")
def text_to_sign(req: TextRequest):

    video_path = generate_video(req.text)

    return FileResponse(
        path=video_path,
        media_type="video/mp4",
        filename="sign.mp4"
    )