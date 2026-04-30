from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Chargement des données
M30 = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\XAUUSD_M30_Features_Final.csv"
H1 = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\XAUUSD_H1_Features_Final.csv"
H4 = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\XAUUSD_H4_Features_Final.csv"

chemins = [M30, H1, H4]

for chemin in chemins:
    df_final = pd.read_csv(chemin, index_col="Datetime", parse_dates=True)

    print("Préparation des ensembles d'entraînement et de test...")

    # A. Sélection des Features (X) et de la Target (y)
    # ATTENTION : On supprime les prix bruts (non stationnaires) pour ne garder que 
    # les rendements, les indicateurs et les variables temporelles. 
    # Pourquoi ??
    colonnes_a_exclure = ['Open', 'High', 'Low', 'Close', 'Target']
    X = df_final.drop(columns=colonnes_a_exclure)
    y = df_final['Target']

    # B. Split Chronologique (80% Train, 20% Test)
    # On ne mélange surtout pas les données (shuffle=False implicite par le découpage d'index)
    split_index = int(len(df_final) * 0.8)

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    print(f"Période d'entraînement : {X_train.index[0]} à {X_train.index[-1]} ({len(X_train)} bougies)")
    print(f"Période de test (Out-of-Sample) : {X_test.index[0]} à {X_test.index[-1]} ({len(X_test)} bougies)")

    # Entrainement
    print("\nEntraînement du modèle Random Forest...")
    # On limite la profondeur (max_depth) pour éviter l'overfitting (surapprentissage)
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # Evaluation
    print("\nÉvaluation sur les données de Test (Futures) :")
    y_pred = rf_model.predict(X_test)

    # A. Métriques de performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy Globale : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("\nRapport de Classification détaillé :")
    print(classification_report(y_test, y_pred))

    # B. Importance des Variables (Feature Importance)
    # Cela nous dira ce qui fait bouger l'or selon l'IA !
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)

    print("\nTop 5 des variables les plus importantes :")
    print(feature_importances.head(7))

    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de Confusion :")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie Valeur')
    plt.title('Matrice de Confusion')
    plt.show()

    # (Optionnel) Affichage graphique de l'importance des variables
    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar')
    plt.title('Importance des Indicateurs pour prédire le XAUUSD (Baseline RF)')
    plt.ylabel('Score d\'importance')
    plt.tight_layout()
    plt.show()

