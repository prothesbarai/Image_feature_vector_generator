# pip install torch
# pip install git+https://github.com/openai/CLIP.git
import torch
import clip
from PIL import Image

device = "cpu"

model, preprocess = clip.load("ViT-B/32", device=device)


def get_embedding(image_path):
    image = Image.open(image_path).convert("RGB")

    # multi-crop (power trick)
    img1 = preprocess(image).unsqueeze(0).to(device)
    img2 = preprocess(image.resize((256, 256))).unsqueeze(0).to(device)

    with torch.no_grad():
        v1 = model.encode_image(img1)
        v2 = model.encode_image(img2)

    # normalize
    v1 = v1 / v1.norm(dim=-1, keepdim=True)
    v2 = v2 / v2.norm(dim=-1, keepdim=True)

    # final powerful vector
    feature_vector = (v1 + v2) / 2

    return feature_vector.squeeze().tolist()


# Here Image Url Path
vec = get_embedding("C:/Users/Prothes/Desktop/visual_search_features/visual_search_backend_go/uploads/search/red_shirt_2.jpg")
print(vec)
print(len(vec))  # usually 512

