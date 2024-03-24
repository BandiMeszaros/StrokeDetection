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

# create base url for the api that is redirected to the /docs endpoint
@app.get("/")
async def root():
    return {"message": "Welcome to Stroke Prediction API"}



@app.post("/v1/predict", description="Returns the prediction of stroke if data is provided.")
async def handle_post_request(file: UploadFile) -> str:

    if _check_file_extension(file):
        data_row = pd.read_csv(file.file)
        prediction, status_code = engine.predict(data_row)
        if status_code == 200:
            return prediction
        else:
            raise HTTPException(status_code=500, detail=prediction)

    else:
        raise HTTPException(status_code=400, detail="File format not supported. Please provide a csv file.")

@app.post("/path/train")
async def handle_train_request():
    # ... your logic here ...
    return {"message": "Train request received!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)