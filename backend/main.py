from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import clip

from starlette.middleware.cors import CORSMiddleware

app = FastAPI()

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

def load_model():
    global model, preprocess
    if model is None:
        model, preprocess = clip.load("RN50", device=device)
    return model, preprocess


def get_embedding_from_image(image: Image.Image):
    model, preprocess = load_model()

    image = image.convert("RGB")
    img = preprocess(image).unsqueeze(0).to(device)

    with torch.no_grad():
        vector = model.encode_image(img)

    vector = vector / vector.norm(dim=-1, keepdim=True)

    return vector.squeeze().tolist()


@app.get("/")
def home():
    return {"status": "API is running 🚀"}


@app.post("/embed-image")
async def embed_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/png", "image/jpeg", "image/jpg"]:
        return {"error": "Only image files allowed"}

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    vector = get_embedding_from_image(image)

    return {
        "embedding": vector,
        "dimension": len(vector)
    }
