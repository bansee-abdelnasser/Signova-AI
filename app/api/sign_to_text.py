from fastapi import APIRouter, UploadFile, File
import tempfile
import os
from app.services.sign_to_text_service import run_pipeline

router = APIRouter()


@router.post("/")
async def sign_to_text(video: UploadFile = File(...)):

    # read file into memory
    contents = await video.read()

    # create a safe temporary file (auto deleted)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(contents)
        temp_path = tmp.name

    try:
        result = run_pipeline(temp_path)
        return result

    finally:
        os.remove(temp_path)