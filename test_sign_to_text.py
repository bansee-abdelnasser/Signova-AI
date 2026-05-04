from app.services.sign_to_text_service import run_pipeline

video_path = "where_trian_arrive.mp4"

result = run_pipeline(video_path)

print("\n======================")
print("FINAL RESULT")
print("======================")

print("GLOSS:")
print(result["gloss"])

print("\nENGLISH:")
print(result["english"])