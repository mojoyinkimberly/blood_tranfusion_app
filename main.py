from fastapi import FastAPI, Query
import joblib
from sklearn.preprocessing import MinMaxScaler
import uvicorn
import pydantic
from pydantic import BaseModel


app = FastAPI()
model = joblib.load("knn_model.jb", "r")

def normalized_data(x):
    norm_scale = MinMaxScaler()
    result = norm_scale.fit_transform(x)
    return result


class Request_Body(BaseModel):
     blood_test1 : int
     blood_test2 : int
     blood_test3 : int
     blood_test4 : int


@app.post('/predict')
def predict(data : Request_Body):
    test_data = [[
        data.blood_test1,
        data.blood_test2,
        data.blood_test3,
        data.blood_test4
    ]]
    scaled = normalized_data(test_data)
    class_idx = app.predict(scaled)[0]
    return { 'class' : model.target_names[class_idx]}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port = 8080)