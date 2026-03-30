from fastapi import FastAPI, File, UploadFile, Form
from inference import predict
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "temp"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/predict")
async def predict_api(file: UploadFile = File(...), question: str = Form(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        answer = predict(file_path, question)

        return {"answer": answer}

    except Exception as e:
        return {"error": str(e)}