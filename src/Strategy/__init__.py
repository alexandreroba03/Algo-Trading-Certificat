import pandas as pd
import numpy as np

"""
==========================================================
STRATEGIE DE TRADING - DETECTION DE TENDANCES & PULLBACKS
==========================================================

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

"""
def analyze_stock_trends(stock_data: pd.DataFrame, index_data: pd.DataFrame, moving_average_period: int = 50, volume_threshold: float = 0.1) -> dict:

    stock_data = stock_data.copy()
    index_data = index_data.copy()

    stock_data['SMA'] = stock_data['Close'].rolling(window=moving_average_period).mean()
    index_data['SMA'] = index_data['Close'].rolling(window=moving_average_period).mean()
    required_cols = ['Close', 'Volume']
    for col in required_cols:
        if col not in stock_data.columns:
            raise ValueError(f"La colonne '{col}' est manquante dans les données de l'action.")

    stock_data['SMA'] = stock_data['Close'].rolling(window=moving_average_period).mean()
    index_data['SMA'] = index_data['Close'].rolling(window=moving_average_period).mean()

    if 'SMA' not in stock_data.columns:
        raise ValueError("La colonne 'SMA' n'a pas pu être calculée.")

    stock_data.dropna(subset=['SMA', 'Close', 'Volume'], inplace=True)
    index_data.dropna(subset=['SMA', 'Close'], inplace=True)

    stock_data.dropna(subset=['SMA', 'Close', 'Volume'], inplace=True)
    index_data.dropna(subset=['SMA', 'Close'], inplace=True)

    if stock_data.empty or index_data.empty:
        return {
            "trend": "Not enough data",
            "volume_confirmation": "N/A"
        }

    if stock_data['Close'].iloc[-1] > stock_data['SMA'].iloc[-1] and index_data['Close'].iloc[-1] > index_data['SMA'].iloc[-1]:
        trend = "Bullish (Upward Trend)"
    elif stock_data['Close'].iloc[-1] < stock_data['SMA'].iloc[-1] and index_data['Close'].iloc[-1] < index_data['SMA'].iloc[-1]:
        trend = "Bearish (Downward Trend)"
    else:
        trend = "Indecisive (No Clear Trend)"

    stock_data['VolumeChange'] = stock_data['Volume'].pct_change()
    if stock_data['VolumeChange'].iloc[-1] > volume_threshold:
        volume_confirmation = "Volume Confirms Trend"
    else:
        volume_confirmation = "Volume Does Not Confirm Trend"

    return {
        "trend": trend,
        "volume_confirmation": volume_confirmation
    }

def detect_sma_pullbacks(df, window=50, threshold=0.01):

    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=window).mean()
    df['distance'] = abs(df['Close'] - df['SMA']) / df['SMA']
    df['pullback_sma'] = (df['distance'] < threshold) & (df['distance'].shift(1) > threshold)

    return df['pullback_sma']

def detect_fibonacci_pullbacks(df, lookback=100):
    df = df.copy()
    swing_high = df['Close'].rolling(window=lookback).max()
    swing_low = df['Close'].rolling(window=lookback).min()

    fib_0_382 = swing_high - 0.382 * (swing_high - swing_low)
    fib_0_5 = swing_high - 0.5 * (swing_high - swing_low)
    fib_0_618 = swing_high - 0.618 * (swing_high - swing_low)
    tol = 0.01

    close = df['Close']
    near_fib = (
        ((abs(close - fib_0_382) / fib_0_382) < tol) |
        ((abs(close - fib_0_5) / fib_0_5) < tol) |
        ((abs(close - fib_0_618) / fib_0_618) < tol)
    )

    return near_fib

def analyze_minor_trend(df):
    df = df.copy()
    df['signal_rsi'] = df['rsi'] < 30
    df['signal_macd'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['signal_bollinger'] = df['Close'] < df['lower_bb']
    df['micro_rebound'] = (df['Close'] > df['lower_bb']) & (df['Close'].shift(1) <= df['lower_bb'].shift(1))
    df['minor_trend_signal'] = (
        df[['signal_rsi', 'signal_macd', 'signal_bollinger', 'micro_rebound']]
        .sum(axis=1) >= 2
    )

    return df

def generate_signals(
    df: pd.DataFrame,
    index_df: pd.DataFrame,
    sma_window: int = 50,
    fib_window: int = 100,
    distance_sma_thresh: float = 0.025,
    fib_tolerance: float = 0.025,
    rsi_buy_thresh: float = 40,
    rsi_sell_thresh: float = 60,
    min_short_term_signals: int = 1,
    min_short_signals: int = 1,
    volume_lookback: int = 20,
    volume_threshold: float = 1.2
) -> pd.DataFrame:
    df = df.copy()
    index_df = index_df.copy()

    # Moyennes mobiles
    sma_label = f"SMA{sma_window}"
    index_sma_label = f"INDEX_SMA{sma_window}"
    df[sma_label] = df['Close'].rolling(window=sma_window).mean()
    index_df[index_sma_label] = index_df['Close'].rolling(window=sma_window).mean()

    # Tendance Dow
    df['dow_trend'] = (df['Close'] > df[sma_label]) & (index_df['Close'] > index_df[index_sma_label])
    df['downtrend'] = (df['Close'] < df[sma_label]) & (index_df['Close'] < index_df[index_sma_label])

    # Pullbacks
    df['distance_sma'] = abs(df['Close'] - df[sma_label]) / df[sma_label]
    df['pullback_sma'] = (df['distance_sma'] < distance_sma_thresh) & (df['distance_sma'].shift(1) > distance_sma_thresh * 0.8)

    swing_high = df['Close'].rolling(window=fib_window).max()
    swing_low = df['Close'].rolling(window=fib_window).min()
    fib_0_382 = swing_high - 0.382 * (swing_high - swing_low)
    fib_0_5 = swing_high - 0.5 * (swing_high - swing_low)
    fib_0_618 = swing_high - 0.618 * (swing_high - swing_low)

    df['pullback_fib'] = (
        ((abs(df['Close'] - fib_0_382) / fib_0_382) < fib_tolerance) |
        ((abs(df['Close'] - fib_0_5) / fib_0_5) < fib_tolerance) |
        ((abs(df['Close'] - fib_0_618) / fib_0_618) < fib_tolerance)
    )

    # Indicateurs court terme
    df['signal_rsi'] = df['rsi'] < rsi_buy_thresh
    df['signal_macd'] = (df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))
    df['signal_bollinger'] = df['Close'] < df['lower_bb']
    df['micro_rebound'] = (df['Close'] > df['lower_bb']) & (df['Close'].shift(1) <= df['lower_bb'].shift(1))
    df['short_term_count'] = df[['signal_rsi', 'signal_macd', 'signal_bollinger', 'micro_rebound']].sum(axis=1)

    # Surachat simplifié (short)
    df['overbought_rsi'] = df['rsi'] > rsi_sell_thresh
    df['overbought_macd'] = (df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))
    df['short_signal_count'] = df[['overbought_rsi', 'overbought_macd']].sum(axis=1)

    # Volume
    df['avg_volume'] = df['Volume'].rolling(window=volume_lookback).mean()
    df['high_volume'] = df['Volume'] > volume_threshold * df['avg_volume']

    # Filtre de volatilité : ATR z-score
    df['tr'] = np.maximum.reduce([
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1)),
        abs(df['Low'] - df['Close'].shift(1))
    ])
    df['atr'] = df['tr'].rolling(window=14).mean()
    df['atr_zscore'] = (df['atr'] - df['atr'].rolling(30).mean()) / df['atr'].rolling(30).std()
    df['volatility_ok'] = df['atr_zscore'] > 0

    df['volatility'] = df['Close'].rolling(window=14).std()
    vol_threshold = df['volatility'].quantile(0.75)
    df['low_volatility'] = df['volatility'] < vol_threshold

    # Momentum / ROC
    df['roc'] = df['Close'].pct_change(periods=5) * 100
    df['momentum_negative'] = df['roc'] < -0.5

    shorts = df[
        df['downtrend'] &
        (df['short_signal_count'] >= min_short_signals) &
        df['momentum_negative'] &
        df['low_volatility']
        ]

    # Candidats
    df['buy_candidate'] = (
        (df['dow_trend'].astype(int) +
         df['pullback_sma'].astype(int) +
         df['pullback_fib'].astype(int) +
         (df['short_term_count'] >= min_short_term_signals).astype(int)) >= 2
    )

    df['sell_candidate'] = (
        (df['rsi'] > rsi_sell_thresh) |
        ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))) |
        (df['Close'] > df['upper_bb'])
    )

    df['short_candidate'] = (
            df['downtrend'] &
            (df['short_signal_count'] >= min_short_signals) &
            (df['momentum_negative'] | (df['roc'] < 0)) &
            (df['low_volatility'] | df['high_volume'])
    )

    df['cover_candidate'] = (
        (df['rsi'] < rsi_buy_thresh) |
        ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))) |
        (df['Close'] < df['lower_bb'])
    )

    # Signaux
    in_position = None
    buy_signals, sell_signals, short_signals = [], [], []

    for i in range(len(df)):
        if df['buy_candidate'].iloc[i] and in_position is None:
            buy_signals.append(True)
            sell_signals.append(False)
            short_signals.append(False)
            in_position = 'long'
        elif df['short_candidate'].iloc[i] and in_position is None:
            buy_signals.append(False)
            sell_signals.append(False)
            short_signals.append(True)
            in_position = 'short'
        elif df['sell_candidate'].iloc[i] and in_position == 'long':
            buy_signals.append(False)
            sell_signals.append(True)
            short_signals.append(False)
            in_position = None
        elif df['cover_candidate'].iloc[i] and in_position == 'short':
            buy_signals.append(False)
            sell_signals.append(True)
            short_signals.append(False)
            in_position = None
        else:
            buy_signals.append(False)
            sell_signals.append(False)
            short_signals.append(False)

    df['buy_signal'] = buy_signals
    df['sell_signal'] = sell_signals
    df['short_signal'] = short_signals

    # Analyse des signaux ignorés
    df['short_ignored'] = df['short_candidate'] & ~df['short_signal']
    ignored_dates = df[df['short_ignored']].index.tolist()

    df.rename(columns={sma_label: 'SMA50'}, inplace=True)

    return df[['Close', 'SMA50', 'rsi', 'macd', 'macd_signal', 'upper_bb', 'lower_bb',
               'buy_candidate', 'sell_candidate', 'short_candidate', 'cover_candidate',
               'buy_signal', 'sell_signal', 'short_signal',
               'dow_trend', 'pullback_sma', 'pullback_fib', 'short_term_count',
               'high_volume']]

def simple_backtest(
    df: pd.DataFrame,
    initial_balance=10000,
    stop_loss_long=-0.05,
    take_profit_long=0.10,
    stop_loss_short=-0.02,
    take_profit_short=0.06,
    trailing_trigger_long=0.05,
    trailing_stop_long=0.02,
    trailing_trigger_short=0.03,
    trailing_stop_short=0.015,
    cooldown_period=5
) -> dict:

    in_position = False
    entry_price, entry_date = 0, None
    position_type = None
    trades, stop_loss_triggers = [], 0
    balance, peak = initial_balance, initial_balance
    max_drawdown = 0
    max_pnl_pct = 0
    cooldown_timer = 0

    for i, (date, row) in enumerate(df.iterrows()):
        price = row['Close']

        if cooldown_timer > 0:
            cooldown_timer -= 1
            continue

        if not in_position:
            if row['buy_signal']:
                in_position = True
                position_type = 'long'
                entry_price, entry_date = price, date
                max_pnl_pct = 0
            elif row['short_signal']:
                in_position = True
                position_type = 'short'
                entry_price, entry_date = price, date
                max_pnl_pct = 0
        else:
            pnl = price - entry_price if position_type == 'long' else entry_price - price
            pct_return = pnl / entry_price
            max_pnl_pct = max(max_pnl_pct, pct_return)
            stop_triggered = False


            if position_type == 'long' and max_pnl_pct > trailing_trigger_long:
                if pct_return <= (max_pnl_pct - trailing_stop_long):
                    stop_triggered = True

            if position_type == 'short' and max_pnl_pct > trailing_trigger_short:
                if pct_return <= (max_pnl_pct - trailing_stop_short):
                    stop_triggered = True


            if position_type == 'long' and row['sell_signal']:
                stop_triggered = True
            if position_type == 'short' and row['cover_candidate']:
                stop_triggered = True

            if stop_triggered:
                if position_type == "long":
                    balance += pct_return * balance
                    if pct_return < 0:
                        stop_loss_triggers += 1
                elif position_type == "short":
                    balance += pct_return * balance
                    if pct_return < 0:
                        stop_loss_triggers += 1

                trades.append({
                    "type": position_type,
                    "date_entry": str(entry_date.date()),
                    "date_exit": str(date.date()),
                    "entry_price": round(entry_price, 2),
                    "exit_price": round(price, 2),
                    "pnl_$": round(pnl, 2),
                    "pct_return": round(pct_return * 100, 2),
                    "duration_days": (date - entry_date).days
                })

                in_position = False
                entry_price, entry_date = 0, None
                position_type = None
                cooldown_timer = cooldown_period


        peak = max(peak, balance)
        drawdown = (peak - balance) / peak
        max_drawdown = max(max_drawdown, drawdown)

    short_trades = [t for t in trades if t["type"] == "short"]
    avg_return = np.mean([t["pct_return"] for t in trades]) if trades else 0
    win_ratio = np.mean([t["pnl_$"] > 0 for t in trades]) * 100 if trades else 0

    return {
        "trades": trades,
        "performance": {
            "starting_balance": initial_balance,
            "ending_balance": round(balance, 2),
            "total_return_%": round((balance - initial_balance) / initial_balance * 100, 2),
            "average_trade_return_%": round(avg_return, 2),
            "win_ratio_%": round(win_ratio, 2),
            "number_of_trades": len(trades),
            "max_drawdown_%": round(max_drawdown * 100, 2),
            "stop_loss_triggers": stop_loss_triggers
        }
    }


def simulate_price_series(mu=0.0005, sigma=0.02, start_date="2020-01-01", end_date="2024-12-31", start_price=100):
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    n_days = len(dates)

    returns = np.random.normal(loc=mu, scale=sigma, size=n_days)
    prices = [start_price]
    for r in returns:
        prices.append(prices[-1] * (1 + r))

    price_series = pd.Series(prices[1:], index=dates, name="Close")
    return price_series

def build_synthetic_dataframe(close_series):
    df = pd.DataFrame()
    df['Close'] = close_series
    df['Open'] = df['Close'].shift(1).fillna(df['Close'])
    df['High'] = df[['Open', 'Close']].max(axis=1) * (1 + np.random.uniform(0, 0.01, len(df)))
    df['Low'] = df[['Open', 'Close']].min(axis=1) * (1 - np.random.uniform(0, 0.01, len(df)))
    df['Volume'] = np.random.randint(1e5, 5e5, size=len(df))
    return df

close_series = simulate_price_series()
df = build_synthetic_dataframe(close_series)

def build_equity_curve(dates, trades, starting_balance=10000):
    balance_series = pd.Series(starting_balance, index=dates, dtype='float64')
    balance = starting_balance

    for trade in trades:
        trade_date = pd.to_datetime(trade["date_exit"])
        if trade_date in balance_series.index:
            balance *= (1 + trade["pct_return"] / 100)
            balance_series.loc[trade_date:] = balance

    return balance_series

def run_multiple_simulations(n_simulations=30, mu_values=None, sigma_values=None):
    results = []

    mu_values = mu_values or np.linspace(0.0003, 0.0012, num=4)
    sigma_values = sigma_values or np.linspace(0.01, 0.03, num=4)

    for mu in mu_values:
        for sigma in sigma_values:
            for _ in range(n_simulations):
                close_series = simulate_price_series(mu=mu, sigma=sigma)
                df_sim = build_synthetic_dataframe(close_series)

                from src.TI import TechInd
                ti = TechInd(df=df_sim)
                df_ti = ti.compute_all()

                df_final = generate_signals(df=df_ti, index_df=df_ti)
                backtest_result = simple_backtest(df_final)

                equity_curve = build_equity_curve(df_sim.index, backtest_result["trades"])

                results.append({
                    "mu": mu,
                    "sigma": sigma,
                    "total_return": backtest_result["performance"]["total_return_%"],
                    "details": backtest_result["performance"],
                    "returns": equity_curve
                })

    return results