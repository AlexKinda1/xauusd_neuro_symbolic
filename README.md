# XAUUSD Neuro Symbolic Trading System

## Description

Un système de trading neuro-symbolique avancé pour XAUUSD (Gold vs US Dollar) qui combine l'apprentissage profond (modèle LSTM) avec l'intelligence artificielle symbolique (agents CrewAI/LangChain). Ce système intègre des analyses prédictives, des évaluations macroéconomiques et une gestion rigoureuse des risques pour prendre des décisions de trading éclairées et automatisées.

## Fonctionnalités principales

### Modèle Prédictif Deep Learning
- Architecture LSTM/GRU pour la prédiction des prix XAUUSD
- Entraînement et validation sur données historiques OHLC
- Sauvegarde et chargement des poids du modèle

###  Agents IA Symboliques
- **Économiste** : Analyse des facteurs macroéconomiques (calendrier économique, indicateurs)
- **Quant** : Analyse statistique et prédictions du modèle
- **Risk Manager** : Évaluation des risques et calcul des stops/take profits
- Orchestration via CrewAI pour une collaboration intelligente

### 📊 Pipeline de Données
- Récupération de données depuis OANDA, MT5 ou yfinance
- Prétraitement : Normalisation MinMaxScaler, fenêtres glissantes
- Ingénierie de features : RSI, MACD via pandas-ta

### 🎛️ Interface Utilisateur
- Dashboard Streamlit interactif
- Visualisation des graphiques, logs des agents et rapports de performance
- Interface intuitive pour surveillance et contrôle

### ⚖️ Gestion des Risques
- Règles strictes : Risque max par trade (1%), perte journalière max (5%)
- Calculs automatiques de Stop-Loss et Take-Profit
- Conformité aux meilleures pratiques de trading

## Installation

### Prérequis
- Python 3.8+
- Git

### Étapes d'installation

1. **Clonez le dépôt** :
   ```bash
   git clone https://github.com/votreusername/xauusd_neuro_symbolic.git
   cd xauusd_neuro_symbolic
   ```

2. **Installez les dépendances** :
   ```bash
   pip install -e .
   ```
   Ou si vous utilisez un environnement virtuel :
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Sur Windows
   pip install -e .
   ```

3. **Configurez les variables d'environnement** :
   Copiez `.env` et remplissez vos clés API :
   ```
   OANDA_API_KEY=votre_clé_oanda
   MT5_LOGIN=votre_login_mt5
   MT5_PASSWORD=votre_mot_de_passe_mt5
   DATABASE_URL=sqlite:///data/trading.db
   ```

## Utilisation

### Lancement de l'application
```bash
python main.py
```
Cela lance le dashboard Streamlit accessible généralement sur `http://localhost:8501`.

### Entraînement du modèle
```python
from src.predictive_model.train import train_model
from src.data_pipeline.preprocessor import create_sequences
import yaml

# Charger les paramètres
with open('config/model_params.yaml') as f:
    params = yaml.safe_load(f)

# Vos données d'entraînement
# X_train, y_train = ...

model = train_model(X_train, y_train, params)
model.save('data/models/lstm_model.h5')
```

### Exécution des agents IA
```python
from src.agents.crew_orchestrator import run_crew

result = run_crew()
print(result)
```

### Exploration de données
Ouvrez `notebooks/01_data_exploration.ipynb` dans Jupyter pour explorer et préparer vos données avant l'implémentation.

## Structure du projet

```
xauusd_neuro_symbolic/
│
├── .env                        # Variables d'environnement (API keys)
├── .gitignore                  # Ignore data/, .env, etc.
├── pyproject.toml              # Dépendances et métadonnées
├── main.py                     # Point d'entrée principal
├── README.md                   # Ce fichier
│
├── config/                     # Configurations
│   ├── model_params.yaml       # Hyperparamètres LSTM
│   ├── agent_prompts.yaml      # Rôles des agents IA
│   └── trading_rules.yaml      # Règles de trading
│
├── data/                       # Données (ignorées par Git)
│   ├── raw/                    # Données brutes OHLC
│   ├── processed/              # Données nettoyées
│   └── models/                 # Poids des modèles entraînés
│
├── src/                        # Code source principal
│   │
│   ├── data_pipeline/          # Pipeline de données
│   │   ├── fetcher.py          # Récupération de données
│   │   ├── preprocessor.py     # Prétraitement et normalisation
│   │   └── features.py         # Calcul des indicateurs techniques
│   │
│   ├── predictive_model/       # Modèle prédictif
│   │   ├── architecture.py     # Définition du modèle LSTM
│   │   ├── train.py            # Script d'entraînement
│   │   └── inference.py        # Prédictions en temps réel
│   │
│   ├── agents/                 # Système d'agents IA
│   │   ├── crew_orchestrator.py# Orchestration des agents
│   │   ├── roles.py            # Définition des rôles
│   │   ├── tasks.py            # Définition des tâches
│   │   └── tools/              # Outils des agents
│   │       ├── dl_tool.py      # Interface avec le modèle DL
│   │       ├── macro_tool.py   # Données macroéconomiques
│   │       └── risk_tool.py    # Calculs de risque
│   │
│   └── ui/                     # Interface utilisateur
│       └── dashboard.py        # Dashboard Streamlit
│
└── notebooks/                  # Environnement de recherche
    └── 01_data_exploration.ipynb # Notebook d'exploration
```

## Dépendances principales

- **pandas** : Manipulation de données
- **numpy** : Calculs numériques
- **tensorflow** : Framework deep learning
- **streamlit** : Interface web
- **crewai** : Framework d'agents IA
- **langchain** : Chaînage d'agents
- **pyyaml** : Gestion des configurations
- **pandas-ta** : Indicateurs techniques
- **yfinance** : Données financières
- **scikit-learn** : Outils ML

Voir `pyproject.toml` pour la liste complète et les versions.

## Configuration

### Paramètres du modèle
Modifiez `config/model_params.yaml` pour ajuster :
- epochs
- batch_size
- seq_length
- learning_rate
- dropout

### Rôles des agents
Personnalisez les prompts dans `config/agent_prompts.yaml`.

### Règles de trading
Ajustez les seuils de risque dans `config/trading_rules.yaml`.

## Développement

### Tests
```bash
# Exécuter les tests (à implémenter)
pytest
```

### Linting
```bash
# Vérifier le code
flake8 src/
black src/
```

### Validation
Après modifications substantielles, exécutez :
```bash
python -m py_compile src/**/*.py
```



### Guidelines
- Suivez PEP 8 pour le style de code
- Ajoutez des tests pour les nouvelles fonctionnalités
- Mettez à jour la documentation
- Respectez les règles de trading définies

## Licence

## Roadmap

- [ ] Intégration temps réel avec brokers
- [ ] Backtesting avancé
- [ ] Optimisation des hyperparamètres
- [ ] Interface mobile
- [ ] Support multi-actifs
- [ ] Intégration avec des APIs économiques externes

---

## Avertissement 
Ce projet est réalisé à des fins de recherche en ingénierie logicielle. Les prédictions et les rapports générés par cette intelligence artificielle ne constituent en aucun cas des conseils d'investissement financier. Le trading sur le Forex et les métaux précieux (XAUUSD) comporte des risques de perte en capital très élevés. Ne tradez jamais avec de l'argent réel sur la base de ce système.