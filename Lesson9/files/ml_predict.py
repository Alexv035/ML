import pandas as pd
from sklearn.metrics import roc_auc_score
import dill
dill._dill._reverse_typemap['ClassType'] = type

X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

with open('ml_pipeline.dill', 'rb') as in_strm:
    final_model= dill.load(in_strm)

predictions = final_model.predict_proba(X_test)
pd.DataFrame({'preds': predictions[:, 1]}).to_csv("test_predictions.csv", index=None)

print(roc_auc_score(y_score=predictions[:, 1][:], y_true=y_test.iloc[:]))