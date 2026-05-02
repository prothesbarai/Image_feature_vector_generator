from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
from torchvision import models, transforms
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

# =========================
# 🔥 LOAD MODEL (VERY LIGHTWEIGHT ~10–15MB)
# =========================
def load_model():
    global model

    if model is None:
        model = models.mobilenet_v3_small(weights="DEFAULT")
        model.classifier = torch.nn.Identity()  # remove classifier → feature vector
        model.eval()

    return model


# =========================
# IMAGE TRANSFORM
# =========================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# =========================
# EMBEDDING FUNCTION
# =========================
def get_embedding(image: Image.Image):
    model = load_model()

    image = image.convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        vec = model(img).squeeze()

    result = vec.flatten().tolist()

    # memory cleanup (important for 512MB)
    del img
    del vec
    gc.collect()

    return result


# =========================
# HEALTH CHECK
# =========================
@app.get("/")
def home():
    return {
        "status": "API is running 🚀",
        "model": "MobileNetV3-Small (30MB alternative to CLIP)"
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

        vector = get_embedding(image)

        return {
            "embedding": vector,
            "dimension": len(vector),
            "model": "mobilenet_v3_small"
        }

    except Exception as e:
        return {"error": str(e)}