from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import clip
import os

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (OK for frontend)
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
# 🔥 LAZY MODEL LOADING
# =========================
def load_model():
    global model, preprocess
    if model is None:
        model, preprocess = clip.load("RN50", device=device)
        model.eval()  # IMPORTANT for inference
    return model, preprocess


# =========================
# 🔥 IMAGE EMBEDDING
# =========================
def get_embedding_from_image(image: Image.Image):
    model, preprocess = load_model()

    image = image.convert("RGB")
    img = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vector = model.encode_image(img)

    vector = vector / vector.norm(dim=-1, keepdim=True)

    return vector.squeeze().cpu().tolist()


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {"status": "API is running 🚀"}


# =========================
# IMAGE API
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
            "dimension": len(vector)
        }

    except Exception as e:
        return {"error": str(e)}
