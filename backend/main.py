# =========================
# 📦 IMPORTS
# =========================
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch

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
# 🤖 GLOBAL MODEL
# =========================
clip_model = None
clip_preprocess = None


# =========================
# 📥 LOAD ONCE (CRITICAL FIX)
# =========================
def load_clip():
    global clip_model, clip_preprocess

    if clip_model is None:
        import clip

        clip_model, clip_preprocess = clip.load(
            "ViT-B/16",
            device=device
        )

        clip_model.eval()
        torch.set_num_threads(1)

    return clip_model, clip_preprocess


# =========================
# 🧠 EMBEDDING (DOUBLE CROP + SAFE)
# =========================
def get_embedding(image: Image.Image):

    model, preprocess = load_clip()

    image = image.convert("RGB")

    # =========================
    # ⚡ SAFE DOUBLE CROP (OPTIMIZED)
    # =========================

    img1 = preprocess(image).unsqueeze(0)
    img2 = preprocess(image.resize((224, 224))).unsqueeze(0)

    with torch.no_grad():
        v1 = model.encode_image(img1)
        v2 = model.encode_image(img2)

    # normalize
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)

    # weighted merge (important for accuracy)
    vec = (v1 * 0.6) + (v2 * 0.4)

    result = vec.squeeze().tolist()

    return result


# =========================
# 🏠 HEALTH
# =========================
@app.get("/")
def home():
    return {
        "status": "RUNNING 🚀",
        "model": "CLIP ViT-B/16 DOUBLE-CROP SAFE",
        "ram": "512MB optimized"
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
            "model": "clip-double-crop-safe"
        }

    except Exception as e:
        return {"error": str(e)}
