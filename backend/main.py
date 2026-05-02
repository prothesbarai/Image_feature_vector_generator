from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import clip
import gc
import os

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

model = None
preprocess = None


# =========================
# 🔥 FORCE LOW MEMORY MODE
# =========================
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"
torch.set_num_threads(1)  # IMPORTANT for 512MB CPU apps


# =========================
# 🔥 LOAD MODEL (LAZY + SAFE)
# =========================
def load_model():
    global model, preprocess

    if model is None:
        # RN50 = lighter than ViT-B/32
        model, preprocess = clip.load("RN50", device="cpu")

        model.eval()

        # extra memory optimization
        for p in model.parameters():
            p.requires_grad = False

    return model, preprocess


# =========================
# 🔥 IMAGE PREPROCESS SAFE
# =========================
def get_embedding_from_image(image: Image.Image):
    model, preprocess = load_model()

    # 🔥 reduce memory usage aggressively
    image = image.convert("RGB")
    image = image.resize((224, 224))  # critical for 512MB

    img = preprocess(image).unsqueeze(0).to("cpu")

    with torch.no_grad():
        vector = model.encode_image(img)

        # normalize (CLIP standard)
        vector = vector / vector.norm(dim=-1, keepdim=True)

    # move back to CPU + free GPU-like buffers
    result = vector.squeeze().cpu().tolist()

    # 🔥 memory cleanup (VERY IMPORTANT on Render)
    del img
    del vector
    gc.collect()

    return result


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "API is running 🚀",
        "model": "CLIP RN50 (512MB optimized)"
    }


# =========================
# IMAGE VECTOR API
# =========================
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Only image files allowed"}

    try:
        image_bytes = await file.read()

        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        vector = get_embedding_from_image(image)

        return {
            "embedding": vector,
            "dimension": len(vector),
            "status": "success"
        }

    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }