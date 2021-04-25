from urllib import request

import pandas as pd
from sklearn.metrics import roc_auc_score

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")



import urllib.request
import json

def get_prediction(age, trtbps, chol, thalachh, oldpeak, caa, sex, cp, fbs, restecg, exng, slp, thall):
    body = {'age': age,
            'Gender of the person': sex,
            'Chest Pain type chest pain type': cp,
            'resting blood pressure (in mm Hg': trtbps,
            'cholestoral in mg/dl fetched via BMI sensor': chol,
            '(fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)': fbs,
            'resting electrocardiographic results': restecg,
            'maximum heart rate achieved': thalachh,
            'exercise induced angina (1 = yes; 0 = no)': exng,
            'Previous peak': oldpeak,
            'Slope': slp,
            'number of major vessels (0-3)': caa,
            'Thal rate': thall
            }

    myurl = "http://0.0.0.0:8180/model"
    req = urllib.request.Request(myurl)
    req.add_header('Content-Type', 'application/json; charset=utf-8')
    jsondata = json.dumps(body)
    jsondataasbytes = jsondata.encode('utf-8')   # needs to be bytes
    req.add_header('Content-Length', len(jsondataasbytes))
    #print (jsondataasbytes)
    response = urllib.request.urlopen(req, jsondataasbytes)
    return json.loads(response.read())['predictions']

get_prediction('Start')

predictions = X_test[['age','trtbps', 'chol', 'thalachh', 'oldpeak', 'caa', 'sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'thall']].iloc[:500].apply(lambda x: get_prediction(x[0],
                                                                                               x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12]), 1)

print(roc_auc_score(y_score=predictions.values, y_true=y_test.iloc[:500]))



