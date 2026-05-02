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
# 🔥 IMPORTANT: MEMORY OPTIMIZATION FLAGS
# =========================
os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"] = "1"


# =========================
# 🔥 LOAD MODEL (LAZY + SAFE)
# =========================
def load_model():
    global model, preprocess

    if model is None:
        model, preprocess = clip.load("RN50", device="cpu")
        model.eval()

    return model, preprocess


# =========================
# 🔥 IMAGE EMBEDDING (512MB SAFE)
# =========================
def get_embedding_from_image(image: Image.Image):
    model, preprocess = load_model()

    # reduce memory usage
    image = image.convert("RGB")
    image = image.resize((224, 224))

    img = preprocess(image).unsqueeze(0).to("cpu")

    with torch.no_grad():
        vector = model.encode_image(img).cpu()

    # normalize
    vector = vector / vector.norm(dim=-1, keepdim=True)

    # cleanup memory
    del img
    gc.collect()

    return vector.squeeze().tolist()


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "API is running 🚀",
        "model": "RN50 (lazy loaded)"
    }


# =========================
# IMAGE API
# =========================
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):

    # validate file type
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Only image files allowed"}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        vector = get_embedding_from_image(image)

        return {
            "embedding": vector,
            "dimension": len(vector)
        }

    except Exception as e:
        return {"error": str(e)}
