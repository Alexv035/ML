import numpy as np
import pandas as pd

import dill

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report

import warnings
warnings.filterwarnings('ignore')

def evaluate_preds(model, X_train, X_test, y_train, y_test):
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    get_classification_report(y_train, y_train_pred, y_test, y_test_pred)

def get_classification_report(y_train_true, y_train_pred, y_test_true, y_test_pred):
    print('TRAIN\n\n' + classification_report(y_train_true, y_train_pred))
    print('TEST\n\n' + classification_report(y_test_true, y_test_pred))
    print('CONFUSION MATRIX\n')
    print(pd.crosstab(y_test_true, y_test_pred))

TRAIN_DATASET_PATH = './heart.csv'

df = pd.read_csv(TRAIN_DATASET_PATH)
TARGET_NAME = 'output'
NUM_FEATURE_NAMES = ['age','trtbps', 'chol', 'thalachh']
LOG_FEATURE_NAMES = ['oldpeak', 'caa']
CAT_FEATURE_NAMES = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'thall']
SELECTED_FEATURE_NAMES = NUM_FEATURE_NAMES + CAT_FEATURE_NAMES +LOG_FEATURE_NAMES

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_norm = df.copy()
df_norm[NUM_FEATURE_NAMES] = scaler.fit_transform(df_norm[NUM_FEATURE_NAMES])
df = df_norm.copy()

data_norm = df.copy()
data_norm[LOG_FEATURE_NAMES] = np.log10(data_norm[LOG_FEATURE_NAMES] + 1)
df = data_norm.copy()

UPDATED_DATASET_PATH = './new_train.csv'
df.to_csv(UPDATED_DATASET_PATH, index=False, encoding='utf-8')

TRAIN_DATASET_PATH = './new_train.csv'
df = pd.read_csv(TRAIN_DATASET_PATH)

NUM_FEATURE_NAMES = ['age','trtbps', 'chol', 'thalachh', 'oldpeak', 'caa']
CAT_FEATURE_NAMES = ['sex', 'cp', 'fbs', 'restecg', 'exng', 'slp', 'thall']
SELECTED_FEATURE_NAMES = NUM_FEATURE_NAMES + CAT_FEATURE_NAMES

X = df[SELECTED_FEATURE_NAMES]
y = df[TARGET_NAME]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, shuffle=True, random_state=21, stratify=y)

#save test
X_test.to_csv("X_test.csv", index=None)
y_test.to_csv("y_test.csv", index=None)
#save train
X_train.to_csv("X_train.csv", index=None)
y_train.to_csv("y_train.csv", index=None)

from catboost import CatBoostClassifier

model_catb = CatBoostClassifier(silent=True, random_state=21,
                                     cat_features=CAT_FEATURE_NAMES,
#                                      one_hot_max_size=7
                                     )
model_catb.fit(X_train, y_train)

disbalance = y_train.value_counts()[0] / y_train.value_counts()[1]

model_catb = CatBoostClassifier(silent=True, random_state=21,
                                     cat_features=CAT_FEATURE_NAMES,
                                     class_weights=[1, disbalance]
                                     )
model_catb.fit(X_train, y_train)

frozen_params = {
     'class_weights':[1, disbalance],
     'silent':True,
     'random_state':21,
     'cat_features':CAT_FEATURE_NAMES,
     'eval_metric':'F1',
     'early_stopping_rounds':20
}
model_catb = CatBoostClassifier(**frozen_params)

params = {'iterations':[100, 200, 500, 700, 1500],
          'max_depth':[3, 5, 7]}

cv = StratifiedKFold(n_splits=3, random_state=21, shuffle=True)

grid_search = model_catb.grid_search(params, X_train, y_train, cv=cv, stratified=True, refit=True)

dpth = grid_search['params']['depth']
iterns = grid_search['params']['iterations']

final_model = CatBoostClassifier(**frozen_params, iterations=iterns, max_depth=dpth)
final_model.fit(X_train, y_train, eval_set=(X_test, y_test))

print(evaluate_preds(final_model, X_train, X_test, y_train, y_test))

with open("ml_pipeline.dill", "wb") as f:
    dill.dump(final_model, f)



