# =========================
# 📦 IMPORTS
# =========================
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import gc

from starlette.middleware.cors import CORSMiddleware

# =========================
# 🚀 APP
# =========================
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
# 🤖 GLOBAL MODEL (KEEP ALIVE)
# =========================
clip_model = None
clip_preprocess = None


# =========================
# 📥 LOAD ONCE ONLY (IMPORTANT FIX)
# =========================
def load_clip():
    global clip_model, clip_preprocess

    if clip_model is None:
        import clip

        clip_model, clip_preprocess = clip.load(
            "ViT-B/16",
            device=device
        )

        # 🔥 memory optimization
        clip_model = clip_model.half()
        clip_model.eval()

        torch.set_num_threads(1)

    return clip_model, clip_preprocess


# =========================
# 🧠 EMBEDDING (OPTIMIZED FOR STABILITY)
# =========================
def get_embedding(image: Image.Image):

    model, preprocess = load_clip()

    image = image.convert("RGB")

    # =========================
    # ⚡ SIMPLE SINGLE-CROP (IMPORTANT FIX)
    # =========================
    img = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vec = model.encode_image(img)

    # normalize
    vec = vec / vec.norm(dim=-1, keepdim=True)

    result = vec.squeeze().tolist()

    # cleanup only temp tensors
    del img, vec
    gc.collect()

    return result


# =========================
# 🏠 HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "RUNNING 🚀",
        "model": "CLIP ViT-B/16 stable mode",
        "memory": "512MB optimized"
    }


# =========================
# 📤 API
# =========================
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):

    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Only image files allowed"}

    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        vector = get_embedding(image)

        return {
            "embedding": vector,
            "dimension": len(vector),
            "model": "clip-stable-512mb"
        }

    except Exception as e:
        return {"error": str(e)}
