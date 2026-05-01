print("STARTING TEST")

from pipelines.text_to_sign.run import run_pipeline

video_path = run_pipeline("Yesterday morning the firefighter arrived at the hospital after the accident and helped the injured people in the emergency room.”")

print("Returned:", video_path)