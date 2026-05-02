# =========================
# 📦 IMPORTS
# =========================
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import torch
import threading

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
# 🤖 GLOBAL MODEL (LAZY LOAD ONLY)
# =========================
model = None
preprocess = None

# 🔥 prevent parallel crash (IMPORTANT FOR RENDER)
lock = threading.Lock()


# =========================
# 📥 LOAD MODEL ONLY WHEN NEEDED
# =========================
def load_clip():
    global model, preprocess

    if model is None:
        import clip

        model, preprocess = clip.load(
            "ViT-B/16",
            device=device
        )

        model.eval()
        torch.set_num_threads(1)


# =========================
# 🧠 EMBEDDING FUNCTION
# =========================
def get_embedding(image: Image.Image):

    load_clip()

    image = image.convert("RGB")

    # ⚡ single safe crop (NO MEMORY SPIKE)
    img = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        vec = model.encode_image(img)

    # normalize (cosine similarity ready)
    vec = vec / vec.norm(dim=-1, keepdim=True)

    return vec.squeeze().tolist()


# =========================
# 🏠 HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "RUNNING 🚀",
        "model": "CLIP ViT-B/16 LAZY LOAD STABLE",
        "ram": "512MB safe mode"
    }


# =========================
# 📤 IMAGE VECTOR API
# =========================
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):

    # 🔥 lock prevents parallel memory crash
    with lock:

        if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
            return {"error": "Only images allowed"}

        try:
            image_bytes = await file.read()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            vector = get_embedding(image)

            return {
                "embedding": vector,
                "dimension": len(vector),
                "model": "clip-vi16-lazy-safe"
            }

        except Exception as e:
            return {"error": str(e)}
