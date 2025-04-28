# STRATEGIE DE TRADING - DETECTION DE TENDANCES & PULLBACKS

Ce fichier implémente une stratégie algorithmique combinant :
- Détection de tendances moyennes/long terme (via moyennes mobiles).
- Identification de pullbacks (corrections de court terme) sur différents niveaux (SMA et Fibonacci).
- Utilisation d'indicateurs techniques classiques (RSI, MACD, Bandes de Bollinger).
- Analyse du volume et de la volatilité pour confirmer les signaux.
- Gestion des entrées/sorties avec stop-loss, take-profit, et trailing stops.

Hypothèses principales :
- Une action qui surperforme sa moyenne mobile (et l'indice de référence) est en tendance haussière.
- Les pullbacks sont des opportunités d'entrée dans une tendance existante (achat ou vente à découvert).
- Les signaux de retournement court terme sont exploités via RSI, MACD, et Bandes de Bollinger.
- Les opérations short sont favorisées uniquement dans des contextes de faible volatilité et de momentum négatif.
- Le volume est utilisé pour confirmer la solidité des mouvements.

Indicateurs utilisés :
- SMA (Simple Moving Average) : pour détecter la tendance générale.
- Pullbacks SMA : distance par rapport à la moyenne mobile pour repérer un retour temporaire.
- Fibonacci retracements : seuils de 38.2%, 50%, 61.8% pour identifier des niveaux de pullback probables.
- RSI : détecter les zones de surachat/survente (achat si <40, vente si >60).
- MACD : croisement MACD/MACD Signal pour valider le momentum.
- Bandes de Bollinger : rebond sur la borne basse pour achats.
- Volume : confirmation des signaux si volume supérieur à la moyenne.
- ATR Z-Score : analyse de la volatilité anormale pour éviter les périodes trop calmes ou trop agitées.
- ROC (Rate of Change) : évaluation du momentum sur 5 jours.

Construction de la stratégie :
- Signaux d'achat :
    - Tendance haussière confirmée (SMA).
    - Pullback détecté sur SMA ou Fibonacci.
    - Renforcement par au moins 1 signal de retournement court terme (RSI, MACD, Bollinger).
- Signaux de vente (long exit) :
    - Retour en surachat ou cassure du momentum positif.
- Signaux de vente à découvert (short) :
    - Tendance baissière.
    - Momentum négatif confirmé (ROC ou variation négative > seuil).
    - Faible volatilité ou forte activité sur volume.
- Signaux de rachat de short (cover) :
    - Conditions inverses du short (retour de la force ou survente détectée).

Gestion des positions :
- Stop-loss et Take-profit dynamique :
    - Basés sur seuils fixes OU trailing stops activés dès que la position est gagnante.
- Période de cooldown obligatoire après chaque trade (évite sur-trading).

Simulation :
- Un module simule des séries de prix aléatoires selon un Geometric Brownian Motion (GBM).
- La stratégie est testée sur ces séries synthétiques sur la période 2020–2024.
- Analyse de la performance moyenne, du meilleur et du pire scénario.

---

## Installation

**Cloner le dépôt GitHub :**

```bash
git clone https://github.com/alexandreroba03/Algo-Trading-Certificat.git
cd Algo-Trading-Certificat