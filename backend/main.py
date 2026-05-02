# =========================
# 📦 IMPORTS
# =========================
from fastapi import FastAPI, UploadFile, File  # API framework
from PIL import Image  # image processing
import io  # byte stream handling
import torch  # ML tensor operations
import gc  # garbage collection (memory cleanup)

from starlette.middleware.cors import CORSMiddleware  # allow frontend calls

# =========================
# 🚀 FASTAPI APP
# =========================
app = FastAPI()

# =========================
# 🌐 CORS SETUP (frontend access allow)
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # allow all frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# ⚙️ DEVICE SETTING (CPU only for Render free tier)
# =========================
device = "cpu"

# =========================
# 🤖 GLOBAL CLIP MODEL VARIABLES (lazy loading)
# =========================
clip_model = None  # CLIP model stored here after first load
clip_preprocess = None  # image preprocessing pipeline


# =========================
# 📥 LOAD CLIP MODEL (only when needed)
# =========================
def load_clip():
    global clip_model, clip_preprocess

    if clip_model is None:
        import clip

        clip_model, clip_preprocess = clip.load(
            "ViT-B/32",
            device="cpu"
        )

        clip_model = clip_model.half()
        clip_model.eval()

        torch.set_num_threads(1)

    return clip_model, clip_preprocess


# =========================
# 🧠 EMBEDDING GENERATION FUNCTION
# =========================
def get_embedding(image: Image.Image):

    # load model (lazy)
    model, preprocess = load_clip()

    # convert image to RGB (safe format)
    image = image.convert("RGB")

    # =========================
    # 🔥 MULTI-CROP TECHNIQUE (accuracy boost)
    # =========================

    # normal resized image
    img1 = preprocess(image).unsqueeze(0)

    # zoomed version (captures more detail)
    img2 = preprocess(image.resize((256, 256))).unsqueeze(0)

    # =========================
    # 🚀 MODEL INFERENCE (no gradient = faster + less RAM)
    # =========================
    with torch.no_grad():

        # encode image → vector representation
        v1 = model.encode_image(img1)
        v2 = model.encode_image(img2)

    # =========================
    # 📏 NORMALIZATION (important for cosine similarity)
    # =========================
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)

    # =========================
    # 🔥 FINAL VECTOR (average of both crops)
    # =========================
    vec = (v1 + v2) / 2

    # convert tensor → python list (for JSON response)
    result = vec.squeeze().tolist()

    # =========================
    # 🧹 MEMORY CLEANUP (VERY IMPORTANT for 512MB)
    # =========================
    del img1, img2, v1, v2, vec  # remove variables
    gc.collect()  # force free RAM

    # unload model to prevent memory leak
    unload_clip()

    return result


# =========================
# 🏠 HEALTH CHECK API
# =========================
@app.get("/")
def home():
    return {
        "status": "API is running 🚀",  # server status
        "model": "CLIP ViT-B/32 (optimized multi-crop version)",  # model info
        "memory": "optimized for 512MB RAM"
    }


# =========================
# 📤 IMAGE VECTOR API
# =========================
@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):

    # =========================
    # ❌ FILE VALIDATION
    # =========================
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Only image files allowed"}

    try:
        # =========================
        # 📥 READ IMAGE FROM REQUEST
        # =========================
        image_bytes = await file.read()  # read raw bytes

        # convert bytes → image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # =========================
        # 🧠 GENERATE VECTOR
        # =========================
        vector = get_embedding(image)

        # =========================
        # 📤 RESPONSE TO FRONTEND
        # =========================
        return {
            "embedding": vector,  # final vector output
            "dimension": len(vector),  # usually 512
            "model": "clip-vit-b32-multicrop-optimized"
        }

    except Exception as e:
        # =========================
        # ❌ ERROR HANDLING
        # =========================
        return {"error": str(e)}
