import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

recall_werte = {
    'GBM': 0.7,
    'GBM (optimiert)': 0.71,
    'XGBoost': 0.7,
    'XGBoost (optimiert)': 0.71,
    'Lasso': 0.67,
}


plt.figure(figsize=(15, 6))
plt.bar(recall_werte.keys(), recall_werte.values(), color=sns.color_palette("Paired"))
plt.xlabel('Modell')
plt.ylabel('Recall')
plt.ylim(0.5, 0.85)
plt.title('Recall-Werte der verschiedenen Modelle')
plt.savefig('Recall_vergleich.pdf',dpi=2000,bbox_inches='tight')
plt.show()
