# 📌 Image Feature Vector Generator
Image_feature_vector_generator হলো একটি lightweight Python-based tool যা যেকোনো image থেকে deep learning model ব্যবহার করে feature vector (embedding) তৈরি করে।
এই feature vector মূলত একটি numerical representation, যা image-এর visual information (shape, color, texture, pattern) কে vector আকারে encode করে।

# 🚀 কী করে এই প্রজেক্ট?
এই tool দিয়ে তুমি সহজেই:
- 🖼️ Image থেকে feature vector generate করতে পারবে
- 🔍 Visual similarity / image search system বানাতে পারবে
- 🤖 Machine Learning / AI model এর জন্য dataset প্রস্তুত করতে পারবে
- 🛒 E-commerce product image matching system তৈরি করতে পারবে
- 📊 Clustering / classification tasks করতে পারবে

# 🧠 কীভাবে কাজ করে?
এই প্রজেক্ট সাধারণত pre-trained deep learning model (যেমন CLIP / CNN / ResNet) ব্যবহার করে:
- Image load করা হয়
- Model image কে process করে
- High-dimensional feature vector generate করে
- সেই vector কে database বা file এ store করা যায়

## 🚀 Install

``` bash
pip install torch
pip install git+https://github.com/openai/CLIP.git
```

## 🚀 Exact Python দিয়ে install

``` bash
C:\Users\Prothes\AppData\Local\Programs\Python\Python39\python.exe -m pip install git+https://github.com/openai/CLIP.git
```

------------------------------------------------------------------------

## 📌 Feature Vector কী?

Feature vector হলো একটি image-এর mathematical representation (সংখ্যার
list), যা AI বুঝতে পারে।

👉 যেমন: একটি "red shirt" image → \[0.12, -0.33, 0.98, ...\]

------------------------------------------------------------------------

## 🧠 Use Cases (কোথায় ব্যবহার হয়)

### 🧾 Product Recommendation System
### 🧠 AI-based Image Understanding
### 📱 Flutter / Web app image matching backend
### 🗂️ Duplicate image detection

### 🛍️ Visual Search (Google Lens style system)

-   image দিয়ে product search (Daraz, Amazon style)

### 🛍️ E-commerce Recommendation

-   similar product দেখানো

### 🧬 Image Similarity

-   duplicate image detect

### 📂 Image Clustering

-   similar image group করা

### 🧾 Content Moderation

-   inappropriate image detect

------------------------------------------------------------------------

## ⚙️ Image থেকে Feature Vector বের করার Example Simple (👉 একবার ছবি দেখে opinion নেয় ) But Fast 🚀 

``` python
import torch
import clip
from PIL import Image

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image = preprocess(Image.open("image.jpg")).unsqueeze(0).to(device)

with torch.no_grad():
    feature = model.encode_image(image)

feature = feature / feature.norm(dim=-1, keepdim=True)

print(feature[0].tolist())
```

------------------------------------------------------------------------

## ⚙️ Image থেকে Feature Vector বের করার Example Multi-crop(👉 দুইটা angle থেকে দেখে final decision নেয়) But Slower 🐢
``` python
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
```

------------------------------------------------------------------------

# এই CLIP embedding কোডটা API বানাতে গেলে মূলত ৩টা জিনিস লাগবে:
- Server framework (API বানানোর জন্য)
- Endpoint (route/controller)
- Image receive + embedding return logic
## ✅ FastAPI দিয়ে API বানানো (BEST & MODERN)
## Install dependency
```bash
pip install fastapi uvicorn python-multipart torch clip-by-openai pillow
```
## 🚀 API কোড
```python
from fastapi import FastAPI, UploadFile, File
import torch
import clip
from PIL import Image
import io

app = FastAPI()

device = "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


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
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes))

    vector = get_embedding_from_image(image)

    return {
        "embedding": vector,
        "dimension": len(vector)
    }
```
------------------------------------------------------------------------
# Need ✅ requirements.txt
```bash
fastapi
uvicorn
torch
ftfy
regex
tqdm
pillow
git+https://github.com/openai/CLIP.git
```
------------------------------------------------------------------------


## 🔥 Feature Vector দিয়ে কী কী করা যায়?

### ✔ Similarity Search

``` python
cos_sim = torch.nn.functional.cosine_similarity(vec1, vec2)
```

### ✔ Recommendation System

-   similar embedding match করে product suggest

### ✔ Vector Database (FAISS)

-   fast search

------------------------------------------------------------------------

## 🧱 System Architecture

Flutter → Go Backend → Python ML → Vector DB → Result

------------------------------------------------------------------------

## 🎯 Conclusion

Feature vector = AI এর চোখ 👁️\
এটা দিয়েই পুরো visual search system কাজ করে।

------------------------------------------------------------------------

## ❤️ Tips

-   Normalization ব্যবহার
-   Multi-image embedding
-   FAISS use করলে speed বাড়বে
------------------------------------------------------------------------

## 📌 Summary
- 👉 এই tool image কে “numbers” এ convert করে
- 👉 তারপর সেই numbers দিয়ে AI বুঝতে পারে কোন image কেমন
- 👉 পুরো system টা modern visual search engine এর backbone