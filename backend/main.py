from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import clip
import gc

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
# 🔥 LOAD MODEL ON STARTUP (IMPORTANT FIX)
# =========================
@app.on_event("startup")
def startup():
    global model, preprocess
    model, preprocess = clip.load("RN50", device="cpu")
    model.eval()


# =========================
# 🔥 SAFE EMBEDDING FUNCTION
# =========================
def get_embedding_from_image(image: Image.Image):
    image = image.convert("RGB")

    # resize to reduce memory spike (IMPORTANT for 512MB)
    image = image.resize((224, 224))

    img = preprocess(image).unsqueeze(0).to("cpu")

    with torch.no_grad():
        vector = model.encode_image(img).cpu()

    # normalize
    vector = vector / vector.norm(dim=-1, keepdim=True)

    # memory cleanup
    del img
    gc.collect()

    return vector.squeeze().tolist()


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
