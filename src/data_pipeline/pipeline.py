import pandas as pd
import numpy as np
import glob
import os
import pandas_ta as ta

def load_and_merge_histdata(folder_path):
    """
    Charge tous les fichiers CSV annuels de HistData d'un dossier,
    les fusionne et crée un index temporel propre.
    """
    # Trouver tous les fichiers CSV dans le dossier
    all_files = glob.glob(os.path.join(folder_path, "*.csv"))
    df_list = []
    
    # Noms des colonnes selon la structure HistData
    col_names = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    for file in all_files:
        print(f"Chargement de {file}...")
        # Chargement sans header
        df_temp = pd.read_csv(file, names=col_names, header=None)
        df_list.append(df_temp)
        
    # Concaténer tous les DataFrames
    df = pd.concat(df_list, ignore_index=True)
    
    # Créer un vrai format Datetime Pandas et le définir comme index
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%Y.%m.%d %H:%M')
    df = df.set_index('Datetime')
    df = df.drop(columns=['Date', 'Time'])
    
    # Trier chronologiquement (très important !)
    df = df.sort_index()
    
    return df

# ==========================================
# 2. RÉÉCHANTILLONNAGE M1 -> H1
# ==========================================
def resample_m1_to_h1(df):
    """
    Convertit les données 1 Minute en bougies 1 Heure.
    """
    print("Rééchantillonnage de M1 vers M30...")
    # Dictionnaire d'agrégation pour reconstruire les bougies H1
    ohlcv_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }
    
    df_h1 = df.resample('30min').agg(ohlcv_dict)
    
    # Supprimer les heures vides (week-ends et jours fériés où le marché est fermé)
    df_h1 = df_h1.dropna()
    
    return df_h1

# ==========================================
# 3. FEATURE ENGINEERING (Ingénierie des variables)
# ==========================================
def engineer_features(df):
    """
    Génère les indicateurs techniques et transforme les données pour le Machine Learning.
    """
    print("Création des Features (Indicateurs techniques et temporels)...")
    
    # A. Rendements logarithmiques (Pour la stationnarité - Crucial pour les LSTM)
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # B. Indicateurs de Tendance (Trend)
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    
    # Distance entre le prix et l'EMA (pour normaliser l'information)
    df['Dist_EMA_20'] = (df['Close'] - df['EMA_20']) / df['EMA_20']
    
    # C. Indicateurs de Momentum
    df['RSI_14'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9) # Moving Average Convergence Divergence (MACD)
    df = pd.concat([df, macd], axis=1) # Ajoute MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    
# D. Volatilité
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    bbands = ta.bbands(df['Close'], length=20, std=2)
    
    # SOLUTION ROBUSTE : On cherche dynamiquement les colonnes contenant 'BBB' (Width) et 'BBP' (Percent)
    col_width = [col for col in bbands.columns if 'BBB' in col][0]
    col_percent = [col for col in bbands.columns if 'BBP' in col][0]
    
    # On assigne les valeurs en utilisant les noms de colonnes trouvés
    df['BB_Width'] = bbands[col_width]
    df['BB_Percent'] = bbands[col_percent]
    
    # E. Encodage Cyclique du Temps (Session de trading)
    df['Hour'] = df.index.hour
    df['DayOfWeek'] = df.index.dayofweek
    
    df['Hour_Sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_Cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    
    # On nettoie les variables brutes d'heures pour ne garder que le signal cyclique
    df = df.drop(columns=['Hour'])
    
    return df

# 4. CRÉATION DE LA TARGET (Classification Binaire)

def create_target(df):
    """
    Définit ce que le modèle doit prédire. 
    1 = Hausse sur la prochaine heure, 0 = Baisse/Stagnation.
    """
    print("Création de la variable cible (Target)...")
    
    # On compare la clôture de T+1 par rapport à la clôture de T
    # Attention au shift(-1) pour regarder dans le futur !
    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    
    # Le dernier record aura un Target 'NaN' car T+1 n'existe pas, on le supprime
    df = df.dropna()
    
    return df

# ==========================================
# EXECUTION DU PIPELINE
# ==========================================
print("Mon dossier de travail actuel est :", os.getcwd())

CHEMIN_DOSSIER = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\csv"

# Petite vérification de sécurité avant de lancer la machine :
fichiers_trouves = glob.glob(os.path.join(CHEMIN_DOSSIER, "*.csv"))
print(f"Nombre de fichiers CSV trouvés : {len(fichiers_trouves)}")

if len(fichiers_trouves) == 0:
    print("ATTENTION : Toujours aucun fichier trouvé. Vérifie le chemin exact dans l'explorateur Windows !")
else:
    # 3. Exécution si les fichiers sont bien là
    print("Fichiers trouvés, lancement du pipeline...")
    df_raw = load_and_merge_histdata(CHEMIN_DOSSIER)
    df_h1 = resample_m1_to_h1(df_raw)
    df_features = engineer_features(df_h1)
    df_final = create_target(df_features)
    
    print("\nAperçu du dataset final prêt pour le ML :")
    print(df_final[['Close', 'Log_Returns', 'RSI_14', 'Hour_Sin', 'Target']].tail())
    print(f"\nTaille finale du dataset : {df_final.shape}")
    
nom_fichier = "XAUUSD_M30_Features_Final.csv"

# On utilise le même dossier de base que tout à l'heure
dossier_sauvegarde = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1"

# Chemin complet
chemin_complet = os.path.join(dossier_sauvegarde, nom_fichier)

# Exportation en CSV
# ATTENTION : index=True est OBLIGATOIRE ici, car tes dates (Datetime) sont stockées dans l'index !
df_final.to_csv(chemin_complet, index=True)

print(f" Dataset final sauvegardé avec succès avec {len(df_final)} lignes et {len(df_final.columns)} colonnes !")
print(f" Emplacement : {chemin_complet}")