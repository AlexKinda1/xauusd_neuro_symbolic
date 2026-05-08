import pandas as pd
import numpy as np
import yfinance as yf
import logging
import glob
import os
import warnings

# Désactiver les avertissements inutiles de Pandas pour garder une console propre
warnings.filterwarnings('ignore')
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

""""
# ==============================================================================
# MODULE 1 : ACQUISITION DE L'HORLOGE MAÎTRE (OHLCV XAUUSD H1)
# ==============================================================================
def load_and_resample_xauusd(folder_path: str) -> pd.DataFrame:
    print(f"[1/4] Chargement des données XAUUSD depuis {folder_path}...")
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if not all_files:
        raise FileNotFoundError("Aucun fichier CSV trouvé. Vérifie ton chemin absolu !")
        
    df_list = []
    col_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for file in all_files:
        df_temp = pd.read_csv(file, names=col_names, header=None)
        df_list.append(df_temp)
        
    df = pd.concat(df_list, ignore_index=True)
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df.set_index('Datetime', inplace=True)
    df.drop(columns=['Date', 'Time'], inplace=True)
    df.sort_index(inplace=True)
    
    # Rééchantillonnage en H1 (Notre Master Clock)
    ohlcv_dict = {'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'}
    df_h1 = df.resample('1h').agg(ohlcv_dict).dropna()
    logger.info(f"      -> Master Clock générée : {len(df_h1)} bougies H1.")
    return df_h1
"""
# ==============================================================================
# MODULE 2 : ACQUISITION DE LA MACROÉCONOMIE (Basse Fréquence)
# ==============================================================================
def fetch_macro_context(start_date: str, end_date: str) -> pd.DataFrame:
    """
    Télécharge les indices macroéconomiques clés (VIX, DXY, US10Y) via Yahoo Finance.
    APPLIQUE UN SHIFT(1) POUR GARANTIR L'ABSENCE DE FUITE DU FUTUR (Data Leakage).
    """
    logger.info("[2/4] Téléchargement des données Macroéconomiques...")
    
    # ^VIX : Volatilité S&P500 | DX-Y.NYB : US Dollar Index | ^TNX : Taux obligations US 10 ans
    tickers = ["^VIX", "DX-Y.NYB", "^TNX"]
    
    # Téléchargement silencieux
    df_macro = yf.download(tickers, start=start_date, end=end_date, progress=False)['Close']
    
    # Renommer les colonnes pour plus de clarté
    df_macro.columns = ['DXY', '^TNX', '^VIX'] # Attention à l'ordre retourné par yfinance
    df_macro.rename(columns={'^TNX': 'US10Y', '^VIX': 'VIX'}, inplace=True)
    
    # SECURITÉ ABSOLUE (Le coeur scientifique du script)
    # Les données téléchargées sont des clôtures journalières.
    # Pour qu'elles soient utilisables le jour J, il faut qu'elles représentent la clôture de J-1.
    df_macro_safe = df_macro.shift(1)
    
    logger.info(f"      -> {len(df_macro_safe)} jours de contexte macro sécurisé générés.")
    return df_macro_safe

# ==============================================================================
# MODULE 3 : FUSION ET ALIGNEMENT TEMPOREL MULTIMODAL (Le Data Lake)
# ==============================================================================
def build_multimodal_datalake(df_xauusd: pd.DataFrame, df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Greffe les données macroéconomiques sur l'horloge maître H1 de l'or.
    Utilise un Forward Fill (ffill) pour propager la donnée basse fréquence sur la haute fréquence.
    """
    logger.info("[3/4] Alignement asynchrone des séries temporelles (Fusion)...")

    # 1. Extraire la date pure de l'horloge H1 pour permettre la jointure avec la macro (qui est en D1)
    df_xauusd['Just_Date'] = df_xauusd.index.normalize()
    
    # 2. Jointure à gauche (Left Join) : On garde chaque heure de l'or, et on y colle la macro du jour
    # Comme df_macro a déjà été "shifté" de 1 jour, l'heure H du mardi recevra bien la macro du lundi soir.
    df_datalake = pd.merge(
        df_xauusd, 
        df_macro, 
        how='left', 
        left_on='Just_Date', 
        right_index=True
    )
    
    # 3. Nettoyage de la colonne de jointure
    df_datalake.drop(columns=['Just_Date'], inplace=True)
    df_datalake.index = df_xauusd.index # Restaurer l'index temporel exact
    
    # 4. Forward Fill strict : S'il y a un jour férié macro (ex: Thanksgiving), 
    # on maintient la dernière valeur connue pour combler les NaN.
    df_datalake.ffill(inplace=True)
    
    # 5. Supprimer les premières lignes qui pourraient contenir des NaN à cause du shift
    df_datalake.dropna(inplace=True)
    
    logger.info("[4/4] ✅ Data Lake Multimodal finalisé !")
    return df_datalake

# ==============================================================================
# EXÉCUTION DU PIPELINE ET GESTION DES CHEMINS
# ==============================================================================
if __name__ == "__main__":
    # Renseigne ton chemin absolu ici
    CHEMIN_DOSSIER_XAUUSD = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\csv"
    
    try:
        # Étape A : Création de la base OHLCV
        df_gold_h1 = load_and_resample_xauusd(CHEMIN_DOSSIER_XAUUSD)
        
        # On extrait les dates dynamiquement pour ne télécharger que la macro nécessaire
        date_debut = df_gold_h1.index.min().strftime('%Y-%m-%d')
        date_fin = df_gold_h1.index.max().strftime('%Y-%m-%d')
        
        # Étape B : Récupération macro sécurisée
        df_macro_safe = fetch_macro_context(start_date=date_debut, end_date=date_fin)
        
        # Étape C : Fusion dans le Data Lake
        datalake_final = build_multimodal_datalake(df_gold_h1, df_macro_safe)
        
        print("\n--- APERÇU DU DATA LAKE ---")
        print(datalake_final[['Close', 'VIX', 'DXY', 'US10Y']].tail(10))
        
        # Sauvegarde (Décommente pour enregistrer)
        # datalake_final.to_csv(r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\Datalake_XAUUSD_Macro.csv")
        
    except Exception as e:
        print(f" Une erreur est survenue : {e}")