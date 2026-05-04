from fastapi import APIRouter, UploadFile, File
import shutil
from app.services.sign_to_text_service import run_pipeline

router = APIRouter()

TEMP_PATH = "temp.mp4"

@router.post("/")
async def sign_to_text(video: UploadFile = File(...)):

    with open(TEMP_PATH, "wb") as f:
        shutil.copyfileobj(video.file, f)

    return run_pipeline(TEMP_PATH)