



# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import JSONResponse
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf

# app = FastAPI()

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )

# MODEL_PATH = "/media/joston-saldanha/New Volume/d drive/chickendisease/saved_models/1"  # Adjust the path to your .h5 model file
# CLASS_NAMES = ["cocci", "healthy", "salmonella"]

# # Load the Keras model
# MODEL = tf.keras.models.load_model(MODEL_PATH)

# @app.get("/ping")
# async def ping():
#     return "Hello, I am alive"

# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         image = read_file_as_image(contents)
#         img_batch = np.expand_dims(image, 0)
        
#         predictions = MODEL.predict(img_batch)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]))
        
#         return JSONResponse({
#             'class': predicted_class,
#             'confidence': confidence
#         })
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)





# from fastapi import FastAPI, File, UploadFile, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from fastapi.responses import HTMLResponse, JSONResponse
# import uvicorn
# import numpy as np
# from io import BytesIO
# from PIL import Image
# import tensorflow as tf
# import os

# app = FastAPI()

# # Configure CORS
# origins = [
#     "http://localhost",
#     "http://localhost:3000",
# ]

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["*"],
# )

# # Path to your model
# MODEL_PATH = "/media/joston-saldanha/New Volume/d drive/chickendisease/saved_models/1"
# CLASS_NAMES = ["cocci", "healthy", "salmonella"]

# # Load the Keras model
# MODEL = tf.keras.models.load_model(MODEL_PATH)

# # Serve the index.html page
# @app.get("/", response_class=HTMLResponse)
# async def read_index():
#     with open("index.html") as f:
#         return f.read()

# # Function to read uploaded image file
# def read_file_as_image(data) -> np.ndarray:
#     image = np.array(Image.open(BytesIO(data)))
#     return image

# # Prediction endpoint
# @app.post("/predict")
# async def predict(file: UploadFile = File(...)):
#     try:
#         contents = await file.read()
#         image = read_file_as_image(contents)
#         img_batch = np.expand_dims(image, 0)
        
#         predictions = MODEL.predict(img_batch)
#         predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
#         confidence = float(np.max(predictions[0]))
        
#         return JSONResponse({
#             'class': predicted_class,
#             'confidence': confidence
#         })
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# # Run the FastAPI application with uvicorn
# if __name__ == "__main__":
#     uvicorn.run(app, host="localhost", port=8000)



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

# Configure CORS
origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:8000",  # Add the origin where your HTML page is hosted
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],  # Or specify specific headers needed, e.g., ['Content-Type']
)

# Path to your model
MODEL_PATH = "/media/joston-saldanha/New Volume/d drive/chickendisease/saved_models/1"
CLASS_NAMES = ["cocci", "healthy", "salmonella"]

# Load the Keras model
MODEL = tf.keras.models.load_model(MODEL_PATH)

# Serve the index.html page
@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html") as f:
        return f.read()

# Function to read uploaded image file
def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Prediction endpoint
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

# Run the FastAPI application with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)




