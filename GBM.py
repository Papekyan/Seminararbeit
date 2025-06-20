# -*- coding: utf-8 -*-


# Risikofaktoren für Kardiovaskuläre Herzkrankheiten

## 0. Import von Bibliotheken und Datenveranschaulichung


import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns #Visualisierung
import matplotlib.pyplot as plt #Visualisierung
from sklearn.model_selection import train_test_split #Split Training- und Testdaten
from sklearn.metrics import recall_score,precision_score, confusion_matrix, classification_report, accuracy_score #Evaluation
from sklearn.ensemble import GradientBoostingClassifier #GBM
from sklearn.model_selection import GridSearchCV #hyperparameter tuning
from xgboost import XGBClassifier #XGBoost
from xgboost import plot_tree
import shap

"""## 1. Datenveranschaulichung"""

df = pd.read_csv('heart_data.csv')

df.info()
df.describe()

df.info(-3)

#Kardinalität der Spalten
print(df.nunique())

"""## 2. Datenaufbereitung"""

#Alter in Jahre umrechnen
df['age_years'] = df ['age']/365

#BMI und nicht benötigte Spalten entfernen
df['male'] = df['gender'].apply(lambda x: 1 if x == 1 else 0)
df['female'] = df['gender'].apply(lambda x: 1 if x == 2 else 0)
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)
df = df.drop(['index','id','height','weight','gender'],axis=1)

df.info()

#Aufteilung der Zielvariable

sns.countplot(x=df['cardio'],palette=sns.color_palette() )
plt.xlabel("Zielwert")
plt.ylabel("Anzahl" )
plt.title("(0 = Keine Krankheit, 1 = Krankheit)")

plt.savefig('Zielwert.pdf',dpi=1500,bbox_inches='tight')


#Aufteilung des Datensatzes in Prädiktoren und Zielwerte
X = df.drop('cardio',axis=1)
Y = df['cardio']

#Überprüfen ob NaN Werte vorhanden sind
X.isnull().sum()

#Daten splitten
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""## 3. Anwendung Gradient Boosting Classifier"""

#Modell trainieren

gbr = GradientBoostingClassifier()
gbr = gbr.fit(X_train, Y_train)

#Trainiertes Modell auf Test Set anwende
Y_pred = gbr.predict(X_test)

#Konfusionsmatrix generieren

recall_GBM = recall_score(Y_test, Y_pred, pos_label=1)
accuracy_GBM = accuracy_score(Y_test, Y_pred)
precision_GBM = precision_score(Y_test, Y_pred, pos_label=1)

print('Konfusionsmatrix GBM:')
cm = confusion_matrix(Y_test, Y_pred, labels=[1, 0])
print(cm)

print('Recall GBM:')
print(round(recall_GBM,2))
print('Accuracy GBM:')
print(round(accuracy_GBM,2))
print('Precision GBM:')
print(round(precision_GBM,2))

"""### 3.1 Hyperparamter Tuning"""

#Parameter festlegen über die varriert werden darf
param_grid = {
    'max_depth': [1,3,5],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'n_estimators': [10, 50, 100, 150, 200, 300, 400, 500],}

gbr_tuning = GridSearchCV(gbr, param_grid, scoring= 'recall', cv=3)
gbr_tuning.fit(X_train, Y_train)

#Parameter und Vorhersage des verbesserten Models
print(gbr_tuning.best_params_)
gbr_tuned = gbr_tuning.best_estimator_
Y_tuned_pred = gbr_tuned.predict(X_test)

#Evaluationsreport und Konfusionsmatrix generieren
recall_GBM_tuned = recall_score(Y_test, Y_tuned_pred, pos_label=1)
accuracy_GBM_tuned = accuracy_score(Y_test, Y_tuned_pred)
precision_GBM_tuned = precision_score(Y_test, Y_tuned_pred, pos_label=1)

print('Konfusionsmatrix:')
cm = confusion_matrix(Y_test, Y_tuned_pred, labels=[1, 0])
print(cm)


çprint('Recall GBM tuned:')
print(round(recall_GBM_tuned,2))
print('Accuracy GBM tuned:')
print(round(accuracy_GBM_tuned,2))
print('Precision GBM tuned:')
print(round(precision_GBM_tuned,2))