from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import clip

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cpu"

# =========================
# 🚀 LOAD MODEL ON START (CRITICAL FIX)
# =========================
clip_model, clip_preprocess = clip.load(
    "ViT-B/16",
    device=device
)

clip_model.eval()
torch.set_num_threads(1)


# =========================
# 🧠 SIMPLE SAFE EMBEDDING
# =========================
def get_embedding(image: Image.Image):

    image = image.convert("RGB")

    # ⚡ ONLY ONE CROP (IMPORTANT FOR STABILITY)
    img = clip_preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vec = clip_model.encode_image(img)

    vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.squeeze().tolist()


# =========================
# 🏠 HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "RUNNING 🚀",
        "model": "CLIP ViT-B/16 STABLE MODE",
        "note": "optimized for Render 512MB"
    }


# =========================
# 📤 API
# =========================
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Only images allowed"}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        vector = get_embedding(image)

        return {
            "embedding": vector,
            "dimension": len(vector),
            "model": "clip-stable-render"
        }

    except Exception as e:
        return {"error": str(e)}
