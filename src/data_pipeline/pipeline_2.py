import pandas as pd
import numpy as np
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

path_gold = "XAUUSD_H1_Features_Final.csv"
df_gold = pd.read_csv(path_gold, index_col='Datetime', parse_dates=True)
df_gold.index = pd.to_datetime(df_gold.index)

path_macro = "macro_data.csv"
df_macro = pd.read_csv(path_macro, index_col=0)

# 1. COERCITION TEMPORELLE : On force en Date. Si c'est du texte ("Ticker"), ça devient NaT
df_macro.index = pd.to_datetime(df_macro.index, errors='coerce')

# 2. NETTOYAGE : On supprime les lignes où l'index est NaT (les lignes parasites)
df_macro = df_macro[df_macro.index.notnull()]

# 3. COERCITION NUMÉRIQUE : On force toutes les données à être des nombres (float)
df_macro = df_macro.apply(pd.to_numeric, errors='coerce')


print(f"Chargement terminé : Or ({df_gold.shape}), Macro ({df_macro.shape})")
print(f"Type index Or : {df_gold.index.dtype} | Type index Macro : {df_macro.index.dtype}")

# Nettoyage 
def flatten_columns(df):
    new_cols = []
    for col in df.columns:
        clean_name = col.replace("(", "").replace(")", "").replace("'", "").replace(", ", "_")
        new_cols.append(clean_name)
    df.columns = new_cols
    return df

df_macro = flatten_columns(df_macro)

# SÉCURITÉ ANTI-FUITE (SHIFT)
df_macro_safe = df_macro.shift(1)

# ==============================================================================
# 3. FUSION MULTIMODALE (MASTER CLOCK ALIGNMENT)
# ==============================================================================
df_gold['Merge_Date'] = df_gold.index.normalize()

df_multimodal = pd.merge(
    df_gold,
    df_macro_safe,
    how='left',
    left_on='Merge_Date',
    right_index=True
)

df_multimodal.drop(columns=['Merge_Date'], inplace=True)
df_multimodal.ffill(inplace=True)
df_multimodal.dropna(inplace=True)

# Affichage de vérification
macro_cols_to_show = [c for c in df_multimodal.columns if 'Close' in c]
print("\n--- Aperçu du Dataset Multimodal Final ---")
print(df_multimodal[macro_cols_to_show].tail(5))

df_multimodal.to_csv("XAUUSD_Multimodal_Data_Lake.csv")
print("\nFusion réussie et Data Lake sauvegardé !")