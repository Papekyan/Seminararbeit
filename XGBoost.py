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
from xgboost import plot_tree #Visualisierung
import shap #SHAP

"""## 1. Datenveranschaulichung"""

df = pd.read_csv('heart_data.csv')

df.info()
df.describe()

df.info(-3)

#Kardinalität der Spalten
print(df.nunique())

"""## 2. Datenaufbereitung"""

#Alter in Jahre umrechnen
df['age'] = df ['age']/365

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

plt.show()

#Aufteilung des Datensatzes in Prädiktoren und Zielwerte
X = df.drop('cardio',axis=1)
Y = df['cardio']

#Überprüfen ob NaN Werte vorhanden sind
X.isnull().sum()

#Daten splitten
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

"""## 4.XGBoost Classifier

"""

XGB = XGBClassifier()
XGB.fit(X_train, Y_train)

Y_pred_XGB = XGB.predict(X_test)

#Evaluationsreport und Konfusionsmatrix generieren
report = classification_report(
    Y_test,
    Y_pred_XGB,
    labels=[1, 0],
    digits=2
)




accuracy_xgb = accuracy_score(Y_test, Y_pred_XGB)
recall_xgb = recall_score(Y_test, Y_pred_XGB, pos_label=1)
precision_xgb = precision_score(Y_test, Y_pred_XGB, pos_label=1)

cm = confusion_matrix(Y_test, Y_pred_XGB, labels=[1, 0])
print('Konfusionsmatrix XGBoost:')
print(cm)

print('Recall XGBoost:')
print(round(recall_xgb,2))

print('Accuracy XGBoost:')
print(round(accuracy_xgb,2))

print('Precision XGBoost:')
print(round(precision_xgb,2))

"""### 4.1 Hyperparameter Tuning"""

param_grid = {
    'max_depth': [1,3,5],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
    'n_estimators': [10, 50, 100, 150, 200, 300, 400, 500],}

XGB_tuning = GridSearchCV(XGB, param_grid, cv=3, scoring='recall', n_jobs=-1)

XGB_tuning.fit(X_train, Y_train)

#Parameter und Vorhersage des verbesserten Models
print(XGB_tuning.best_params_)
XGB_tuned = XGB_tuning.best_estimator_
Y_tuned_pred_XGB = XGB_tuned.predict(X_test)

#Evaluationsreport und Konfusionsmatrix generieren

accuracy_tuned_xgb = accuracy_score(Y_test, Y_tuned_pred_XGB)
recall_tuned_xgb = recall_score(Y_test, Y_tuned_pred_XGB, pos_label=1)
precision_tuned_xgb = precision_score(Y_test, Y_tuned_pred_XGB, pos_label=1)


cm = confusion_matrix(Y_test, Y_tuned_pred_XGB, labels=[1, 0])
print('Konfusionsmatrix XGBoost tuned:')
print(cm)

# Konfusionsmatrix für das optimierte XGBoost-Modell plotten
#plt.figure(figsize=(10, 7))
#sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Krankheit', 'Keine Krankheit'], yticklabels=['Krankheit', 'Keine Krankheit'])
#plt.title('Konfusionsmatrix - Optimiertes XGBoost Modell')
#plt.ylabel('Wahre Klasse')
#plt.xlabel('Vorhergesagte Klasse')
#plt.show()

print('Recall tuned XGBoost:')
print(round(recall_tuned_xgb,2))

print('Accuracy tuned XGBoost:')
print(round(accuracy_tuned_xgb,2))

print('Precision tuned XGBoost:')
print(round(precision_tuned_xgb,2))

explainer = shap.Explainer(XGB_tuned, X_train)  # X_train is your feature set
shap_values = explainer(X_train)

# Erstellt einen SHAP Summary Plot

shap.summary_plot(shap_values, X_train, cmap=plt.get_cmap("crest"))

plt.savefig('SHAP.pdf',dpi=2000,bbox_inches='tight')



"""### 4.2 Merkmalswichtigkeit (Feature Importance) mit XGBoost"""

# Merkmalswichtigkeit als Balkendiagramm plotten
# Zeigt die Wichtigkeit der Merkmale basierend auf der Häufigkeit ihrer Verwendung in den Bäumen

plt.figure(figsize=(10, 8))
xgb.plot_importance(XGB_tuned, max_num_features=12, importance_type='weight')
plt.title('Merkmalswichtigkeit XGB')
plt.xlabel('F-Score')
plt.ylabel('Merkmale')
plt.grid(False)  # Entfernt das Raster vom Plot
plt.savefig('Merkmalswichtigkeit.pdf',dpi=2000,bbox_inches='tight')
plt.show()


