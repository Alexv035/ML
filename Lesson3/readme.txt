Model tree
              precision    recall  f1-score   support

           0       0.75      0.40      0.52     10506
           1       0.59      0.87      0.70     10494

    accuracy                           0.63     21000
   macro avg       0.67      0.63      0.61     21000
weighted avg       0.67      0.63      0.61     21000

Model LGBM
              precision    recall  f1-score   support

           0       0.85      0.29      0.44     10506
           1       0.57      0.95      0.71     10494

    accuracy                           0.62     21000
   macro avg       0.71      0.62      0.58     21000
weighted avg       0.71      0.62      0.58     21000

Model Linear regression
MAE 0.4403076666480033, R2 0.12084476520826548

Модель деревья показала лучший результат из 3

ROC подходят, когда наблюдения сбалансированы между каждым классом, тогда как precision_recall_curve подходят для несбалансированных наборов данных.
