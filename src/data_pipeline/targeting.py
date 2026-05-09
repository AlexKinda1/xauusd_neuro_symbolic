import pandas as pd
import numpy as np

print("Étape 2 : Création de l'architecture Double Target...")

# 1. Chargement du Data Lake sécurisé (celui qu'on vient de créer)
df_multimodal = pd.read_csv("XAUUSD_Multimodal_Data_Lake.csv", index_col=0, parse_dates=True)

"""
print(df_multimodal.info())
print(df_multimodal.head())
print(df_multimodal.describe())

"""

# 2. TARGET DE RÉGRESSION (Amplitude du mouvement futur)
# On calcule le rendement logarithmique entre la clôture actuelle (T) et la future clôture (T+1)
# Le shift(-1) permet de regarder "dans le futur" pour créer la cible d'apprentissage
df_multimodal['Target_Reg'] = np.log(df_multimodal['Close_x'].shift(-1) / df_multimodal['Close_x'])

# 3. TARGET DE CLASSIFICATION (Direction future)
# 1 si le rendement futur est strictement positif, 0 sinon.
df_multimodal['Target_Class'] = (df_multimodal['Target_Reg'] > 0).astype(int)

# 4. NETTOYAGE
# La toute dernière ligne du dataset aura des Targets 'NaN' car la bougie T+1 n'existe pas encore.
df_multimodal.dropna(inplace=True)

# 5. SAUVEGARDE FINALE AVANT APPRENTISSAGE
df_multimodal.to_csv("XAUUSD_MachineLearning_Ready.csv")

print("\n--- Aperçu des Targets Créées ---")
print(df_multimodal[['Close_x', 'Target_Reg', 'Target_Class']].tail())
print(f"\n Dataset prêt pour les modèles ML/DL sauvegardé ({df_multimodal.shape[0]} lignes).")
