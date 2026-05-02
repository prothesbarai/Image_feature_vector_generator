from fastapi import FastAPI, UploadFile, File
import torch
import clip
from PIL import Image
import io

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
#model, preprocess = clip.load("ViT-B/32", device=device)
model, preprocess = clip.load("RN50", device=device)


def get_embedding_from_image(image: Image.Image):
    image = image.convert("RGB")

    img1 = preprocess(image).unsqueeze(0).to(device)
    img2 = preprocess(image.resize((256, 256))).unsqueeze(0).to(device)

    with torch.no_grad():
        v1 = model.encode_image(img1)
        v2 = model.encode_image(img2)

    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)

    feature_vector = (v1 + v2) / 2

    return feature_vector.squeeze().tolist()


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
