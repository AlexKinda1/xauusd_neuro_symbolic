import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

print("Préparation des données pour le LSTM...")

H1 = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\XAUUSD_H1_Features_Final.csv"

df_final = pd.read_csv(H1, index_col="Datetime", parse_dates=True)

# ==========================================
# 1. SPLIT TEMPOREL (Train / Validation / Test)
# ==========================================
# Exclure les colonnes non stationnaires
X = df_final.drop(columns=['Open', 'High', 'Low', 'Close', 'Target'])
y = df_final['Target']

# On utilise 70% Train, 15% Validation (pour l'Early Stopping), 15% Test
train_idx = int(len(df_final) * 0.70)
val_idx = int(len(df_final) * 0.85)

X_train_raw, y_train = X.iloc[:train_idx], y.iloc[:train_idx]
X_val_raw, y_val = X.iloc[train_idx:val_idx], y.iloc[train_idx:val_idx]
X_test_raw, y_test = X.iloc[val_idx:], y.iloc[val_idx:]

# ==========================================
# 2. NORMALISATION (SCALING)
# ==========================================
scaler = StandardScaler()
# On apprend la distribution UNIQUEMENT sur le train !
X_train_scaled = scaler.fit_transform(X_train_raw)
X_val_scaled = scaler.transform(X_val_raw)
X_test_scaled = scaler.transform(X_test_raw)

# ==========================================
# 3. CRÉATION DES SÉQUENCES 3D (FENÊTRES GLISSANTES)
# ==========================================
def create_sequences(X_data, y_data, time_steps):
    Xs, ys = [], []
    for i in range(len(X_data) - time_steps):
        Xs.append(X_data[i:(i + time_steps)])
        ys.append(y_data.iloc[i + time_steps]) # Prédiction de la bougie suivante
    return np.array(Xs), np.array(ys)

TIME_STEPS = 24 # On donne au LSTM les 24 dernières heures pour prendre sa décision

X_train, y_train_seq = create_sequences(X_train_scaled, y_train, TIME_STEPS)
X_val, y_val_seq = create_sequences(X_val_scaled, y_val, TIME_STEPS)
X_test, y_test_seq = create_sequences(X_test_scaled, y_test, TIME_STEPS)

print(f"Shape X_train 3D : {X_train.shape} -> (Échantillons, Time Steps, Features)")

# ==========================================
# 4. ARCHITECTURE DU MODÈLE LSTM
# ==========================================
print("\nConstruction du modèle LSTM...")

model = Sequential()
# Couche LSTM : return_sequences=False car on ne veut qu'une seule prédiction à la fin des 24h
model.add(LSTM(units=50, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
# Dropout pour éviter le surapprentissage (éteint 20% des neurones aléatoirement)
model.add(Dropout(0.2))
# Couche de sortie binaire (0 ou 1)
model.add(Dense(units=1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# ==========================================
# 5. ENTRAÎNEMENT AVEC EARLY STOPPING
# ==========================================
# Le fameux set de Validation prend tout son sens ici : 
# Si la perte de validation ne s'améliore plus pendant 5 itérations (patience), on arrête tout !
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Entraînement en cours (ça peut prendre quelques minutes)...")
history = model.fit(
    X_train, y_train_seq,
    epochs=50, # Nombre maximum de passages sur le dataset complet
    batch_size=64,
    validation_data=(X_val, y_val_seq),
    callbacks=[early_stop],
    verbose=1
)

# ==========================================
# 6. ÉVALUATION FINALE (TEST SET)
# ==========================================
print("\nÉvaluation sur les données de Test (Futures) :")
# Le modèle sort des probabilités (ex: 0.8), on les transforme en classes (0 ou 1)
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.7).astype(int)

accuracy = accuracy_score(y_test_seq, y_pred)
print(f" Accuracy LSTM : {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\n Rapport de Classification détaillé :")
print(classification_report(y_test_seq, y_pred))

# Affichage de la courbe d'apprentissage (Loss)
plt.plot(history.history['loss'], label='Loss Entraînement')
plt.plot(history.history['val_loss'], label='Loss Validation')
plt.title('Courbe de Loss du LSTM (Détection Overfitting)')
plt.legend()
plt.show()