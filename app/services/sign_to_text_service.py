import shutil
from pipelines.sign_to_text.run import run_pipeline

TEMP_PATH = "temp/video.mp4"

async def process_video(video):
    with open(TEMP_PATH, "wb") as buffer:
        shutil.copyfileobj(video.file, buffer)

    result = run_pipeline(TEMP_PATH)

    return {
        "gloss": result["gloss"],
        "english": result["english"]
    }