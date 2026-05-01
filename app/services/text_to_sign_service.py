from pipelines.text_to_sign.run import run_pipeline

from app.models.i3d_model import I3DModel  # (optional future use)
from app.core.config import CLASS_LIST


def generate_video(text):
    return run_pipeline(text)