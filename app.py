from fastapi import FastAPI, HTTPException, UploadFile
import uvicorn
import pandas as pd
from MLengine import engine
API_VERSION = "v1"
app = FastAPI(title="Stroke Prediction API",
              version=API_VERSION,
              description="A simple api for stroke detection using binary classification")


def _check_file_extension(file):
    return file.filename.lower().endswith(".csv")


@app.post(f"/{API_VERSION}/predict", description="Returns the prediction of stroke if data is provided.")
async def handle_predict_request(file: UploadFile) -> dict:

    if _check_file_extension(file):
        data_row = pd.read_csv(file.file)
        prediction, status_code = engine.predict(data_row)
        if status_code == 200:
            return prediction
        else:
            raise HTTPException(status_code=500, detail=prediction)

    else:
        raise HTTPException(status_code=400, detail="File format not supported. Please provide a csv file.")


@app.post(f"/{API_VERSION}/train", description="Trains the model with the provided data.")
async def handle_train_request(train_file: UploadFile):
    
    if _check_file_extension(train_file):
        data = pd.read_csv(train_file.file)
        train, status_code = engine.train(data)
        if status_code == 200:
            return train
        else:
            raise HTTPException(status_code=500, detail=train)
    else:
        raise HTTPException(status_code=400, detail="File format not supported. Please provide a csv file.")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
