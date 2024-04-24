import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Daten einlesen
df = pd.read_csv("Stammdaten.csv", low_memory=False)

print(f"Max value: {min(df['Gesamtwert'])}")
print(f"Min value: {max(df['Gesamtwert'])}")

# Diagramm erstellen
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Boxplot für die Spalte "Gesamtwert"
sns.kdeplot(data=df["Gesamtwert"], color='skyblue', fill=True, ax=axs[0])
# sns.kdeplot(x=df["Gesamtwert"], ax=axs[0])
axs[0].set_title("Densityplot Gesamtwert")

# Säulendiagramm für die Spalte "Preiseinheit"
sns.countplot(x=df["Preiseinheit"], ax=axs[1])
axs[1].set_title("Countplot Preiseinheit")

# Layout anpassen
plt.tight_layout()

# Diagramm anzeigen
plt.show()


# Mittelwert der Spalte 'Gesamtwert'
mean_value = df['Gesamtwert'].mean()

# Median der Spalte 'Gesamtwert'
median_value = df['Gesamtwert'].median()

# Quartile der Spalte 'Gesamtwert'
q1 = df['Gesamtwert'].quantile(0.25)  # 25th percentile
q2 = df['Gesamtwert'].quantile(0.50)  # 50th percentile (median)
q3 = df['Gesamtwert'].quantile(0.75)  # 75th percentile

print("Mittelwert:", mean_value)
print("Median:", median_value)
print("1. Quartil (25th Percentile):", q1)
print("2. Quartil (50th Percentile):", q2)
print("3. Quartil (75th Percentile):", q3)

print(df["Preiseinheit"].value_counts())



