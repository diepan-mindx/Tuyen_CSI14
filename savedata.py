import pandas as pd
import csv
def save_data(age,sex,dataset,cp,trestbps,chol,fbs,restecg,thalch,exang,oldpeak,slope,ca,thal,num):
    df = pd.read_csv("./SPCK/userinput.csv")
    id = "U"+ str(len(df) + 1)
    data = {
        "id": [id],
        "age": [age],
        "sex": [sex],
        "dataset": [dataset],
        "cp": [cp],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [fbs],
        "restecg": [restecg],
        "thalch": [thalch],
        "exang": [exang],
        "oldpeak": [oldpeak],
        "slope": [slope],
        "ca": [ca],
        "thal": [thal],
        "num": [num]
    }
    df = pd.DataFrame(data)
    df.to_csv("userinput.csv", mode="a", header=False, index=False)
    print("Data saved successfully.")