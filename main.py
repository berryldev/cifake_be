import os
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from model_loader import predict_image, get_model, get_model_error, MODEL_PATH
from gradcam import generate_gradcam_overlay_base64

DEPLOY_VERSION = "v1.0.2"

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_model()
    yield

app = FastAPI(
    title="CIFAKE Detection API",
    version=DEPLOY_VERSION,
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def validate_image_file(file: UploadFile):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Nama file tidak ditemukan.")
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format file '{ext}' tidak didukung. Unggah citra JPG, JPEG, PNG, atau WEBP."
        )

@app.get("/")
def root():
    model = get_model()
    return {
        "status": "online",
        "service": "CIFAKE AI Image Detection API",
        "deploy_version": DEPLOY_VERSION,
        "model_loaded": model is not None,
        "model_path": MODEL_PATH,
        "model_error": get_model_error(),
        "endpoints": {
            "health": "/health",
            "predict": "POST /predict",
            "explain": "POST /explain"
        }
    }

@app.get("/health")
def health_check():
    model = get_model()
    return {
        "status": "healthy",
        "deploy_version": DEPLOY_VERSION,
        "model_status": "ready" if model is not None else "simulated_mode",
        "model_file_exists": os.path.exists(MODEL_PATH),
        "model_error": get_model_error()
    }

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    validate_image_file(file)
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File kosong.")

    try:
        result = predict_image(content)
        result["filename"] = file.filename
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses prediksi: {str(e)}"
        )

@app.post("/explain")
async def explain_endpoint(file: UploadFile = File(...)):
    validate_image_file(file)
    content = await file.read()
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File kosong.")

    try:
        pred = predict_image(content)
        model = get_model()
        overlay_b64 = generate_gradcam_overlay_base64(content, model=model)

        return JSONResponse(content={
            "filename": file.filename,
            "prediction": pred,
            "gradcam_overlay_base64": overlay_b64
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Terjadi kesalahan saat memproses visualisasi XAI: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
