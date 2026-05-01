from fastapi import FastAPI
from app.api import sign_to_text, text_to_sign
from fastapi.staticfiles import StaticFiles

app = FastAPI(title="Signova API")

app.include_router(sign_to_text.router, prefix="/sign-to-text")
app.include_router(text_to_sign.router, prefix="/text-to-sign")

@app.get("/")
def home():
    return {"message": "Signova API running"}
