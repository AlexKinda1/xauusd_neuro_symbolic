import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CHARGEMENT
# ==============================================================================
df = pd.read_csv("XAUUSD_MachineLearning_Ready.csv", index_col=0, parse_dates=True)
df.rename(columns={'Open_x': 'Open', 'High_x': 'High', 'Low_x': 'Low'}, inplace=True)
df = df.sort_index()

close  = df['Close']
high   = df['High']
low    = df['Low']
volume = df.get('Volume_x', pd.Series(np.nan, index=df.index))

# ==============================================================================
# 2. AUDIT COMPLET DES FEATURES — Aucune feature ne doit regarder dans le futur
# ==============================================================================
def audit_features(X, y, label=""):
    """
    Vérifie qu'aucune feature n'est une fuite en mesurant la corrélation
    instantanée de chaque feature avec la cible future.
    Une corrélation > 0.3 à t=0 est suspecte.
    """
    print(f"\n{'─'*55}")
    print(f"AUDIT FEATURES — {label}")
    print(f"{'─'*55}")
    suspects = []
    for col in X.columns:
        r, _ = spearmanr(X[col].fillna(0), y)
        if abs(r) > 0.15:
            suspects.append((col, r))
            print(f"  ⚠ {col:<30} Spearman = {r:+.4f}  ← SUSPECT")
    if not suspects:
        print("  ✓ Aucune feature suspecte (corrélations < 0.15)")
    return suspects

# ==============================================================================
# 3. CONSTRUCTION DES FEATURES — TOUTES ANCRÉES AU PASSÉ
# ==============================================================================
feat = pd.DataFrame(index=df.index)

# Rendements passés (stationnaires, ancrés à t)
for lag in [1, 2, 4, 8, 16, 24, 48]:
    feat[f'ret_lag{lag}'] = close.pct_change(lag)

# Momentum (différence de rendements sur fenêtres)
feat['mom_4_vs_16']  = close.pct_change(4)  - close.pct_change(16)
feat['mom_8_vs_48']  = close.pct_change(8)  - close.pct_change(48)

# Volatilité réalisée (fenêtres passées)
feat['vol_20']  = close.pct_change().rolling(20).std()
feat['vol_100'] = close.pct_change().rolling(100).std()
feat['vol_regime'] = feat['vol_20'] / feat['vol_100'].clip(lower=1e-8)

# Distance aux extremes récents (ancré à t)
feat['dist_high20'] = high.rolling(20).max() / close - 1
feat['dist_low20']  = close / low.rolling(20).min() - 1
feat['dist_high5']  = high.rolling(5).max()  / close - 1
feat['dist_low5']   = close / low.rolling(5).min() - 1

# Indicateurs techniques DÉJÀ dans le dataset (vérifiés stationnaires)
cols_tech_ok = ['RSI_14', 'BB_Percent', 'BB_Width', 'MACDh_12_26_9',
                'Dist_EMA_20', 'ATR_14', 'Hour_Sin', 'Hour_Cos',
                'Log_Returns']
for c in cols_tech_ok:
    if c in df.columns:
        feat[c] = df[c]

# ATR normalisé
feat['ATR_norm'] = feat['ATR_14'] / close if 'ATR_14' in feat.columns else \
                   (high - low).rolling(14).mean() / close

# NE PAS inclure : Volume_x (nan), Close_y, Low.X, High.X (prix bruts)
# NE PAS inclure : ret_8h futur, pct_change(-N)
feat.dropna(inplace=True)

print(f"Features construites : {feat.shape[1]} colonnes")
print(f"Période : {feat.index[0].date()} → {feat.index[-1].date()}")

# ==============================================================================
# 4. CIBLE FUTURE CORRECTE — Vérification formelle
# ==============================================================================
HORIZON = 8

ret_futur = (close.shift(-HORIZON) - close) / close

assert ret_futur.iloc[-1] != ret_futur.iloc[-1], \
    "ERREUR : ret_futur ne contient pas de NaN → formule incorrecte"
print(f"\n✓ Formule ret_futur validée — NaN sur les {HORIZON} dernières lignes")

seuil_up   = ret_futur.quantile(0.80)
seuil_down = ret_futur.quantile(0.20)
print(f"Seuil haussier > {seuil_up*100:.3f}% | Seuil baissier < {seuil_down*100:.3f}%")

# Aligner les deux séries sur l'index COMMUN avant tout filtrage
idx_commun     = feat.index.intersection(ret_futur.dropna().index)
feat_aligned   = feat.loc[idx_commun]
ret_aligned    = ret_futur.loc[idx_commun]

# Cible extrêmes sur l'index commun
target_extreme = pd.Series(np.nan, index=idx_commun)
target_extreme[ret_aligned >= seuil_up]   = 1
target_extreme[ret_aligned <= seuil_down] = 0

# Filtrer les cas extrêmes
mask = target_extreme.notna()
X    = feat_aligned.loc[mask]
y    = target_extreme.loc[mask]

print(f"Cas extrêmes retenus : {len(X):,} / {len(df):,} "
      f"({len(X)/len(df)*100:.1f}%)")
print(f"Distribution : 1={y.mean():.3f} | 0={1-y.mean():.3f}")

# ==============================================================================
# 5. AUDIT ANTI-FUITE AVANT TOUT ENTRAÎNEMENT
# ==============================================================================
suspects = audit_features(X, y, "Features vs Target FUTUR 8H")

if any(abs(r) > 0.30 for _, r in suspects):
    print("\n🛑 FUITE DÉTECTÉE — Corriger avant d'entraîner")
    print("   Features à supprimer :", [c for c, r in suspects if abs(r) > 0.30])
    # Suppression automatique des features trop corrélées à la cible
    cols_a_supprimer = [c for c, r in suspects if abs(r) > 0.30]
    X = X.drop(columns=cols_a_supprimer, errors='ignore')
    print(f"   → Features restantes : {X.shape[1]}")
else:
    print("\n✓ Audit passé — aucune fuite détectée")

# ==============================================================================
# 6. SPLIT TEMPOREL STRICT
# ==============================================================================
n = len(X)
i1, i2 = int(n * 0.70), int(n * 0.80)

X_tr,  X_val,  X_te  = X.iloc[:i1],  X.iloc[i1:i2],  X.iloc[i2:]
y_tr,  y_val,  y_te  = y.iloc[:i1],  y.iloc[i1:i2],  y.iloc[i2:]

print(f"\nTrain : {X_tr.index[0].date()} → {X_tr.index[-1].date()} ({len(X_tr):,})")
print(f"Val   : {X_val.index[0].date()} → {X_val.index[-1].date()} ({len(X_val):,})")
print(f"Test  : {X_te.index[0].date()} → {X_te.index[-1].date()} ({len(X_te):,})")

# ==============================================================================
# 7. ENTRAÎNEMENT — Walk-Forward avec early stopping sur validation
# ==============================================================================
ratio = (y_tr == 0).sum() / (y_tr == 1).sum()

model = XGBClassifier(
    n_estimators=1000,       # haut → early stopping arrêtera au bon moment
    max_depth=3,
    learning_rate=0.01,
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=50,
    reg_alpha=1.0,
    reg_lambda=3.0,
    scale_pos_weight=ratio,
    random_state=42,
    n_jobs=-1,
    eval_metric='auc',
    early_stopping_rounds=50,  # stoppe si AUC val ne s'améliore pas sur 50 rounds
)

model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False
)

best_iter = model.best_iteration
print(f"\nEarly stopping : meilleure itération = {best_iter}")

# ==============================================================================
# 8. VÉRIFICATION FINALE : L'AUC DOIT ÊTRE < 1.0
# ==============================================================================
proba_te  = model.predict_proba(X_te)[:, 1]
proba_tr  = model.predict_proba(X_tr)[:, 1]
proba_val = model.predict_proba(X_val)[:, 1]

auc_tr  = roc_auc_score(y_tr,  proba_tr)
auc_val = roc_auc_score(y_val, proba_val)
auc_te  = roc_auc_score(y_te,  proba_te)

print(f"\n{'─'*55}")
print(f"AUC Train : {auc_tr:.4f}  {'⚠ overfit sévère' if auc_tr > 0.75 else '✓'}")
print(f"AUC Val   : {auc_val:.4f}  {'⚠ dégradé' if auc_val < auc_tr - 0.05 else '✓'}")
print(f"AUC Test  : {auc_te:.4f}  "
      f"{'✓✓ exploitable' if auc_te > 0.56 else '✓ signal faible' if auc_te > 0.52 else '❌ bruit'}")

if auc_te > 0.99:
    print("\n🛑 AUC encore = 1.0 → fuite résiduelle non résolue")
    print("   Lancer : audit_features(X_te, y_te) pour identifier la source")
elif auc_te > 0.54:
    print("\n✓ Signal réel détecté — passer à l'optimisation")
else:
    print("\n⚠ Signal insuffisant avec les seuls technicals H1")
    print("  → Prochaine étape : ajouter DXY, US10Y réels, VIX, COT Gold")

# Distribution des probabilités (doit avoir de la variance réelle)
print(f"\nDistribution proba test :")
print(f"  min={proba_te.min():.3f} | p25={np.percentile(proba_te,25):.3f} | "
      f"med={np.median(proba_te):.3f} | p75={np.percentile(proba_te,75):.3f} | "
      f"max={proba_te.max():.3f} | std={proba_te.std():.3f}")
print(f"  → Variance utile si std > 0.05 et min < 0.40 et max > 0.60")

# ==============================================================================
# 9. RÉSULTATS AVEC SEUIL OPTIMAL
# ==============================================================================
best_f1, best_thresh = 0, 0.5
for t in np.arange(0.35, 0.70, 0.01):
    f1 = f1_score(y_val, (proba_val > t).astype(int), zero_division=0)
    if f1 > best_f1:
        best_f1, best_thresh = f1, t

pred_te = (proba_te > best_thresh).astype(int)
print(f"\nSeuil (val) : {best_thresh:.2f}")
print(classification_report(y_te, pred_te))

# ==============================================================================
# 10. FEATURE IMPORTANCE POST-AUDIT
# ==============================================================================
print("Top 10 features par importance (XGBoost gain) :")
fi = pd.Series(model.feature_importances_, index=X_tr.columns)
fi = fi.sort_values(ascending=False)
for feat_name, imp in fi.head(10).items():
    corr_val = abs(spearmanr(X_te[feat_name].fillna(0), y_te)[0])
    print(f"  {feat_name:<30} gain={imp:.4f}  corr_test={corr_val:.4f}")

import pandas as pd
import numpy as np
import os
import warnings
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import yfinance as yf
warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CHARGEMENT XAUUSD + MACRO (cache existant réutilisé)
# ==============================================================================
df = pd.read_csv("XAUUSD_MachineLearning_Ready.csv",
                 index_col=0, parse_dates=True).sort_index()
df.rename(columns={'Open_x': 'Open', 'High_x': 'High', 'Low_x': 'Low'}, inplace=True)

close = df['Close']
high  = df['High']
low   = df['Low']

CACHE_FILE = "macro_data_cache.csv"
df_macro_daily = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
print(f"✓ Macro daily chargée : {df_macro_daily.shape}")

# ==============================================================================
# 2. TRANSFORMATION MACRO — TOUT EN STATIONNAIRE
# ==============================================================================
# Règle absolue : aucune feature en niveau absolu (VIX=40, US10Y=4.5, etc.)
# Ces niveaux sont valides dans leur propre régime mais pas cross-régimes.
# On ne garde que des transformations stationnaires.

def build_macro_stationary(df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Transforme toutes les séries macro en features stationnaires :
    - Rendements sur différentes fenêtres (déjà calculés)
    - Z-score rolling (centré sur la moyenne mobile → drift retiré)
    - Signal de tendance (au-dessus/en-dessous de la MA longue)
    - Momentum cross-fenêtres
    Aucun niveau absolu n'est conservé.
    """
    m = pd.DataFrame(index=df_macro.index)

    for asset in ['DXY', 'VIX', 'US10Y', 'SP500', 'GVZ']:
        cl = f'{asset}_close'
        if cl not in df_macro.columns:
            continue

        price = df_macro[cl]

        # Rendements (déjà stationnaires)
        m[f'{asset}_r1d']  = price.pct_change(1)
        m[f'{asset}_r5d']  = price.pct_change(5)
        m[f'{asset}_r20d'] = price.pct_change(20)

        # Z-score rolling 63 jours (≈ 1 trimestre) — retire le niveau absolu
        roll_mean = price.rolling(63).mean()
        roll_std  = price.rolling(63).std().clip(lower=1e-8)
        m[f'{asset}_zscore63'] = (price - roll_mean) / roll_std

        # Signal tendance : +1 si au-dessus MA50, -1 en dessous (stationnaire car borné)
        ma50 = price.rolling(50).mean()
        m[f'{asset}_trend'] = np.sign(price - ma50)

        # Momentum : ret5d minus ret20d (accélération/décélération)
        m[f'{asset}_mom'] = price.pct_change(5) - price.pct_change(20)

    # Features cross-assets (relations entre marchés)
    if 'DXY_r1d' in m.columns and 'SP500_r1d' in m.columns:
        # Corrélation glissante DXY/SP500 sur 20 jours (encode le régime risk-on/off)
        m['DXY_SP500_corr20'] = (m['DXY_r1d'].rolling(20)
                                  .corr(m['SP500_r1d']))

    if 'VIX_r1d' in m.columns and 'SP500_r1d' in m.columns:
        # VIX et SP500 en opposition → encode le stress de marché
        m['fear_spread'] = m['VIX_r1d'] - m['SP500_r1d']

    if 'DXY_r1d' in m.columns and 'US10Y_r1d' in m.columns:
        # Pression combinée sur l'or (DXY monte + taux montent = double pression baissière)
        m['gold_headwind'] = m['DXY_r1d'] + m['US10Y_r1d']

    return m

df_macro_stat = build_macro_stationary(df_macro_daily)
print(f"Features macro stationnaires : {df_macro_stat.shape[1]} colonnes")

# Vérification de stationnarité : toutes les corrélations avec le temps doivent être < 0.1
time_idx = np.arange(len(df_macro_stat))
non_stat = []
for col in df_macro_stat.columns:
    series = df_macro_stat[col].dropna()
    if len(series) < 100:
        continue
    r, _ = spearmanr(np.arange(len(series)), series)
    if abs(r) > 0.15:
        non_stat.append((col, r))

if non_stat:
    print(f"  ⚠ Features encore non-stationnaires :")
    for col, r in non_stat:
        print(f"    {col:<30} corr_time={r:+.3f}  ← à retirer")
else:
    print("  ✓ Toutes les features macro sont stationnaires")

# ==============================================================================
# 3. PROPAGATION DAILY → HORAIRE avec décalage anti-lookahead
# ==============================================================================
def macro_to_hourly_safe(df_macro: pd.DataFrame,
                          hourly_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Propage les données macro daily vers un index horaire.
    Décalage de 1 jour pour que la valeur de J ne soit visible qu'à partir de J+1.
    """
    df_shifted = df_macro.shift(1)   # anti-lookahead : clôture J visible à J+1
    combined   = df_shifted.reindex(df_shifted.index.union(hourly_index)).ffill()
    return combined.reindex(hourly_index)

df_macro_h = macro_to_hourly_safe(df_macro_stat, df.index)
print(f"Macro horaire : {df_macro_h.shape} | NaN total : {df_macro_h.isna().sum().sum()}")

# ==============================================================================
# 4. CONSTRUCTION FEATURES COMPLÈTES
# ==============================================================================
feat = pd.DataFrame(index=df.index)

# ── Technicals H1 (inchangés et validés) ─────────────────────────────────────
for lag in [1, 2, 4, 8, 16, 24, 48]:
    feat[f'ret_lag{lag}'] = close.pct_change(lag)

feat['mom_4_vs_16'] = close.pct_change(4) - close.pct_change(16)
feat['mom_8_vs_48'] = close.pct_change(8) - close.pct_change(48)
feat['vol_20']      = close.pct_change().rolling(20).std()
feat['vol_regime']  = feat['vol_20'] / close.pct_change().rolling(100).std().clip(1e-8)
feat['dist_high20'] = high.rolling(20).max() / close - 1
feat['dist_low20']  = close / low.rolling(20).min() - 1

for c in ['RSI_14', 'BB_Percent', 'BB_Width', 'MACDh_12_26_9',
          'Dist_EMA_20', 'ATR_14', 'Hour_Sin', 'Hour_Cos', 'Log_Returns']:
    if c in df.columns:
        feat[c] = df[c]

# ── Macro stationnaire (nouvelles features corrigées) ────────────────────────
# On exclut les z-scores 63j car 63 jours daily → propager sur 24h crée
# de faux "blocs" de valeur identique. On garde uniquement les rendements
# et les features cross-assets qui changent chaque jour.
cols_macro_utiles = [c for c in df_macro_h.columns
                     if any(c.endswith(s) for s in
                            ['_r1d', '_r5d', '_r20d', '_trend', '_mom',
                             'corr20', 'fear_spread', 'gold_headwind'])]

for col in cols_macro_utiles:
    feat[col] = df_macro_h[col]

feat.dropna(inplace=True)
n_macro = sum(1 for c in feat.columns if any(
    c.startswith(a) for a in ['DXY','VIX','US10Y','SP500','GVZ','fear','gold']))
print(f"\nFeatures totales : {feat.shape[1]}  (tech={feat.shape[1]-n_macro}, macro={n_macro})")

# ==============================================================================
# 5. CIBLE + AUDIT ANTI-FUITE
# ==============================================================================
HORIZON = 8
ret_futur  = (close.shift(-HORIZON) - close) / close
seuil_up   = ret_futur.quantile(0.80)
seuil_down = ret_futur.quantile(0.20)

idx_commun   = feat.index.intersection(ret_futur.dropna().index)
feat_aligned = feat.loc[idx_commun]
ret_aligned  = ret_futur.loc[idx_commun]

target = pd.Series(np.nan, index=idx_commun)
target[ret_aligned >= seuil_up]   = 1
target[ret_aligned <= seuil_down] = 0

mask = target.notna()
X    = feat_aligned.loc[mask]
y    = target.loc[mask]

print(f"Cas extrêmes : {len(X):,} | Distribution 1={y.mean():.3f}")

print(f"\n{'─'*55}")
print("AUDIT ANTI-FUITE")
suspects = []
for col in X.columns:
    r, _ = spearmanr(X[col].fillna(0), y)
    if abs(r) > 0.15:
        suspects.append((col, r))
        print(f"  ⚠ {col:<35} Spearman = {r:+.4f}")

cols_drop = [c for c, r in suspects if abs(r) > 0.30]
if cols_drop:
    X = X.drop(columns=cols_drop)
    print(f"  🛑 Fuite supprimée : {cols_drop}")
elif not suspects:
    print("  ✓ Aucune fuite détectée")

# ==============================================================================
# 6. SPLIT + ENTRAÎNEMENT
# ==============================================================================
n = len(X)
i1, i2 = int(n * 0.70), int(n * 0.80)
X_tr, X_val, X_te = X.iloc[:i1], X.iloc[i1:i2], X.iloc[i2:]
y_tr, y_val, y_te = y.iloc[:i1], y.iloc[i1:i2], y.iloc[i2:]

print(f"\nTrain {len(X_tr):,} | Val {len(X_val):,} | Test {len(X_te):,}")

model = XGBClassifier(
    n_estimators=1000, max_depth=3, learning_rate=0.01,
    subsample=0.7, colsample_bytree=0.7, min_child_weight=50,
    reg_alpha=1.0, reg_lambda=3.0,
    scale_pos_weight=(y_tr==0).sum()/(y_tr==1).sum(),
    random_state=42, n_jobs=-1, eval_metric='auc',
    early_stopping_rounds=50,
)
model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

# ==============================================================================
# 7. RÉSULTATS COMPARATIFS
# ==============================================================================
auc_tr  = roc_auc_score(y_tr,  model.predict_proba(X_tr)[:, 1])
auc_val = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
proba_te = model.predict_proba(X_te)[:, 1]
auc_te  = roc_auc_score(y_te, proba_te)

print(f"\n{'═'*60}")
print(f"{'':32} {'BASELINE':>8} {'MACRO_V1':>9} {'MACRO_V2':>9}")
print(f"{'─'*60}")
print(f"{'AUC Train':<32} {'0.5908':>8} {'0.5918':>9} {auc_tr:>9.4f}")
print(f"{'AUC Val':<32} {'0.5076':>8} {'0.5094':>9} {auc_val:>9.4f}")
print(f"{'AUC Test':<32} {'0.5328':>8} {'0.5260':>9} {auc_te:>9.4f}")
print(f"{'Std proba test':<32} {'0.013':>8} {'0.001':>9} {proba_te.std():>9.3f}")
print(f"{'Early stop iter':<32} {'41':>8} {'4':>9} {model.best_iteration:>9}")
print(f"{'═'*60}")

vs_baseline = auc_te - 0.5328
vs_v1       = auc_te - 0.5260
print(f"\nvs Baseline : {vs_baseline:+.4f} | vs Macro_V1 : {vs_v1:+.4f}")

# ==============================================================================
# 8. APPORT DE CHAQUE FEATURE MACRO
# ==============================================================================
fi = pd.Series(model.feature_importances_, index=X_tr.columns).sort_values(ascending=False)
print("\nTop 15 features :")
print(f"  {'Feature':<35} {'Gain':>7}  {'Corr_test':>10}  Cat")
print(f"  {'─'*60}")
for fn, imp in fi.head(15).items():
    r   = abs(spearmanr(X_te[fn].fillna(0), y_te)[0])
    cat = "MACRO" if any(fn.startswith(a) or fn in ['fear_spread','gold_headwind']
                         for a in ['DXY','VIX','US10Y','SP500','GVZ']) else "tech"
    flag = "  ← utile" if r > 0.03 and cat == "MACRO" else ""
    print(f"  {fn:<35} {imp:>7.4f}  {r:>10.4f}  {cat}{flag}")