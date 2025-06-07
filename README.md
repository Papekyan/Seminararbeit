# Seminararbeit zu GBM und XGBoost
Risikofaktoren für Kardiovaskuläre Herzkrankheiten
Dieses Projekt analysiert die Risikofaktoren für kardiovaskuläre Herzkrankheiten anhand eines Datensatzes und verwendet verschiedene Machine-Learning-Modelle, um Vorhersagen zu treffen.
Inhalt
Datenaufbereitung: Die Daten werden bereinigt, neue Features wie BMI und Alter in Jahren berechnet und unnötige Spalten entfernt.
Datenvisualisierung: Die Zielvariable (Vorhandensein einer Herzkrankheit) wird grafisch dargestellt.
Modellierung: Es werden zwei Modelle verwendet:
Gradient Boosting Classifier
XGBoost Classifier


Hyperparameter-Tuning: Für beide Modelle wird eine Optimierung der Parameter durchgeführt.
Evaluation: Die Modelle werden mit Metriken wie Recall, Precision, F1-Score und einer Konfusionsmatrix bewertet.


Benötigte Bibliotheken
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap



Nutzung
Notebook öffnen: Öffne die Datei Datensatz.ipynb in Jupyter Notebook oder Google Colab.
Daten einlesen: Stelle sicher, dass die Datei heart_data.csv im gleichen Verzeichnis liegt.
Zellen ausführen: Führe die Zellen der Reihe nach aus, um die Daten zu laden, zu verarbeiten, Modelle zu trainieren und die Ergebnisse zu sehen.
Die Visualisierungen helfen, die Verteilung der Daten und die Ergebnisse der Modelle besser zu verstehen.


Das Ziel dieses Projekts ist es, die wichtigsten Risikofaktoren für Herzkrankheiten zu identifizieren und maschinelles Lernen zur Vorhersage einzusetzen. 



