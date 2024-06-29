from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn
import numpy as np
from io import BytesIO
from PIL import Image
import tensorflow as tf
import os

app = FastAPI()


origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    # allow_methods=["*"] for all methods
    allow_methods=["GET", "POST"],
    allow_headers=["*"], 
)


MODEL_PATH='chicken.h5'
CLASS_NAMES = ["coccidiosis", "healthy", "salmonella"]


MODEL = tf.keras.models.load_model(MODEL_PATH)


@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html") as f:
        return f.read()


def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data)).convert("RGB")
    image = image.resize((224, 224))  
    return np.array(image)


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = read_file_as_image(contents)
        img_batch = np.expand_dims(image, 0)
        
        predictions = MODEL.predict(img_batch)
        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))
        
        return JSONResponse({
            'class': predicted_class,
            'confidence': confidence
        })
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
