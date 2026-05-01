from fastapi import APIRouter, UploadFile, File
from app.services.sign_to_text_service import process_video

router = APIRouter()

@router.post("/")
async def sign_to_text(video: UploadFile = File(...)):
    return await process_video(video)