import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# Chemins d'accès aux fichiers

M30 = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\XAUUSD_M30_Features_Final.csv"
H1 = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\XAUUSD_H1_Features_Final.csv"
H4 = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\XAUUSD_H4_Features_Final.csv"

#chemins = [M30, H1, H4]

chemins = [H1] # Visualisation des resultats pour 1H

for chemin in chemins:
    df_final = pd.read_csv(chemin, index_col="Datetime", parse_dates=True)
    
    # ==========================================
    # 1. SÉPARATION DES DONNÉES (SPLIT CHRONOLOGIQUE STRICT)
    # ==========================================
    colonnes_a_exclure = ['Open', 'High', 'Low', 'Close', 'Target']
    X = df_final.drop(columns=colonnes_a_exclure)
    y = df_final['Target']

    split_index = int(len(df_final) * 0.8)

    # Le Test Set reste complètement verrouillé, on ne l'utilise QUE pour l'évaluation finale
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # ==========================================
    # 2. CONFIGURATION DE LA VALIDATION CROISÉE TEMPORELLE
    # ==========================================
    # On crée 5 fenêtres temporelles glissantes pour valider nos hyperparamètres
    tscv = TimeSeriesSplit(n_splits=10)

    # ==========================================
    # 3. DÉFINITION DU MODÈLE ET OPTIMISATION
    # ==========================================
    # XGBoost est surpuissant mais a tendance à overfitter. On va chercher à le brider.
    xgb_model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )

    # Grille de recherche des hyperparamètres
    param_distributions = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5], # On garde une profondeur faible pour le bruit financier
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9], # % de lignes utilisées par arbre (évite le surapprentissage)
        'colsample_bytree': [0.7, 0.8, 0.9] # % de colonnes utilisées par arbre
    }

    print("Recherche des meilleurs paramètres via Walk-Forward Validation (ça peut prendre 1 à 2 minutes)...")
    random_search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_distributions,
        n_iter=10, # Teste 10 combinaisons au hasard. Augmente à 20 si tu as une bonne machine.
        cv=tscv, # LA MAGIE EST ICI : Le modèle respecte la flèche du temps
        scoring='accuracy',
        random_state=42,
        n_jobs=-1,
        verbose=1
    )

    random_search.fit(X_train, y_train)

    print("\n Meilleurs hyperparamètres trouvés :")
    for param, value in random_search.best_params_.items():
        print(f" - {param}: {value}")

    # ==========================================
    # 4. ÉVALUATION FINALE SUR LE TEST SET (LE FUTUR)
    # ==========================================
    # On récupère le meilleur modèle trouvé par la validation
    best_xgb = random_search.best_estimator_

    print("\nÉvaluation du meilleur XGBoost sur les données de Test (Out-of-Sample) :")
    y_pred = best_xgb.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f" Accuracy Globale : {accuracy:.4f} ({accuracy * 100:.2f}%)")

    print("\n Rapport de Classification détaillé :")
    print(classification_report(y_test, y_pred))

    # ==========================================
    # 5. EXPLICABILITÉ (FEATURE IMPORTANCE XGBOOST)
    # ==========================================
    feature_importances = pd.Series(best_xgb.feature_importances_, index=X.columns)
    feature_importances = feature_importances.sort_values(ascending=False)
    
    cm = confusion_matrix(y_test, y_pred)
    print("\nMatrice de Confusion :")
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Prédiction')
    plt.ylabel('Vraie Valeur')
    plt.title('Matrice de Confusion')
    plt.show()

    plt.figure(figsize=(10, 6))
    feature_importances.plot(kind='bar', color='darkred', edgecolor='black')
    plt.title('Importance des Indicateurs pour prédire le XAUUSD (XGBoost Optimisé)')
    plt.ylabel('Score d\'importance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()