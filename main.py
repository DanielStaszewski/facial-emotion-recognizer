from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageOps
from io import BytesIO
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

model = load_model('my_model3.h5')
app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:4200",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def load_image_into_numpy_array(data):
    img = Image.open(data)
    img = img.resize((48,48))
    gray_image = ImageOps.grayscale(img)
    return np.array(gray_image)

@app.post("/files/")
async def create_file(myFile: UploadFile = Form()):
    image = load_image_into_numpy_array(BytesIO(await myFile.read()))
    predictions = model.predict(np.reshape(image, (1, 48, 48)))
    return label_map[predictions[0].tolist().index(max(predictions[0]))]