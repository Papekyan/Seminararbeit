# Risikofaktoren für Kardiovaskuläre Herzkrankheiten - Lasso Regression
# Lasso Regression mit entsprechender Skalierung und automatischer Feature-Selektion

# 0. Import von Bibliotheken
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings('ignore')

# 1. Datenveranschaulichung
print("=== DATENVERANSCHAULICHUNG ===")
df = pd.read_csv('heart_data.csv')

print("Datensatz Info:")
print(df.info())
print("\nDeskriptive Statistiken:")  
print(df.describe())

print(f"\nAnzahl eindeutiger Werte pro Spalte:")
print(df.nunique())

# 2. Datenaufbereitung
print("\n=== DATENAUFBEREITUNG ===")

# Alter in Jahre umrechnen (war in Tagen)
df['age_years'] = df['age'] / 365

# BMI berechnen aus Größe und Gewicht
df['bmi'] = df['weight'] / ((df['height'] / 100) ** 2)

# One-Hot-Encoding für kategoriale Variablen (gluc und cholesterol)
# Diese sind ordinale kategoriale Variablen: 1=normal, 2=über normal, 3=weit über normal
print("One-Hot-Encoding für gluc und cholesterol...")
df = pd.get_dummies(df, columns=['gluc', 'cholesterol'], drop_first=False)

# Umbenennung der OHE-Spalten für bessere Verständlichkeit
df.rename(columns={
    'gluc_1': 'gluc_normal',
    'gluc_2': 'gluc_above_normal', 
    'gluc_3': 'gluc_well_above_normal',
    'cholesterol_1': 'cholesterol_normal',
    'cholesterol_2': 'cholesterol_above_normal',
    'cholesterol_3': 'cholesterol_well_above_normal'
}, inplace=True)

# Entfernen irrelevanter Spalten (Index, ID, height, weight, original age)
columns_to_drop = ['index', 'id', 'height', 'weight', 'age']
# Prüfen welche Spalten tatsächlich existieren
existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
df = df.drop(existing_columns_to_drop, axis=1)

print(f"Finale Datensatz-Form: {df.shape}")
print(f"Spalten nach Aufbereitung: {list(df.columns)}")

# Überprüfung auf fehlende Werte
print(f"\nFehlende Werte: {df.isnull().sum().sum()}")

# 3. Zielvariable und Features definieren
X = df.drop('cardio', axis=1)
y = df['cardio']

print(f"\nFeatures (X): {X.shape}")
print(f"Zielvariable (y): {y.shape}")
print(f"Verteilung Zielvariable:\n{y.value_counts()}")

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrainingsdaten: {X_train.shape}")
print(f"Testdaten: {X_test.shape}")


print("Lasso Regression benötigt skalierte Features für faire Feature-Selektion...")




# 6. Logistische Regression mit Lasso-Regularisierung
print("\n=== LOGISTISCHE REGRESSION MIT LASSO-REGULARISIERUNG ===")

# Logistische Regression mit Lasso-Regularisierung (L1-Penalty)
logistic_lasso_base = LogisticRegression(
    penalty='l1',           # L1-Regularisierung (Lasso) für Feature-Selektion
    solver='liblinear',     # Solver der L1-Penalty unterstützt
    C=1.0,                  # Inverse Regularisierungsstärke (größer = weniger Regularisierung)
    random_state=42,
    max_iter=2000
)

# Modell trainieren
logistic_lasso_base.fit(X_train_scaled, y_train)

# Vorhersagen (automatisch binär bei LogisticRegression!)
y_pred_train_logistic = logistic_lasso_base.predict(X_train_scaled)
y_pred_test_logistic = logistic_lasso_base.predict(X_test_scaled)

# Wahrscheinlichkeiten für detailliertere Analyse
y_pred_proba_train = logistic_lasso_base.predict_proba(X_train_scaled)[:, 1]
y_pred_proba_test = logistic_lasso_base.predict_proba(X_test_scaled)[:, 1]

print("=== EVALUATION LOGISTISCHE REGRESSION BASIS-MODELL ===")
print(f"Training Genauigkeit: {accuracy_score(y_train, y_pred_train_logistic):.4f}")
print(f"Test Genauigkeit: {accuracy_score(y_test, y_pred_test_logistic):.4f}")
print(f"Training Präzision: {precision_score(y_train, y_pred_train_logistic):.4f}")
print(f"Test Präzision: {precision_score(y_test, y_pred_test_logistic):.4f}")
print(f"Training Sensitivität: {recall_score(y_train, y_pred_train_logistic):.4f}")
print(f"Test Sensitivität: {recall_score(y_test, y_pred_test_logistic):.4f}")
print(f"Training F1-Score: {f1_score(y_train, y_pred_train_logistic):.4f}")
print(f"Test F1-Score: {f1_score(y_test, y_pred_test_logistic):.4f}")

# Feature-Selektion anzeigen (Lasso-spezifisch!)
selected_features_logistic = X.columns[logistic_lasso_base.coef_[0] != 0]
eliminated_features_logistic = X.columns[logistic_lasso_base.coef_[0] == 0]
print(f"\nFeature-Selektion (C={logistic_lasso_base.C}):")
print(f"Ausgewählte Features: {len(selected_features_logistic)}/{len(X.columns)}")
print(f"Eliminierte Features: {len(eliminated_features_logistic)}")
if len(eliminated_features_logistic) > 0:
    print(f"Eliminierte Features: {list(eliminated_features_logistic)}")

