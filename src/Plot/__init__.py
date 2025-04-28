import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches

def plot_dow_trend(df, title="Dow Theory Trend"):
    df = df.copy()
    df['SMA'] = df['Close'].rolling(window=50).mean()
    df['Bullish'] = df['Close'] > df['SMA']
    df['Bearish'] = df['Close'] < df['SMA']

    buy_signals = (df['Close'] > df['SMA']) & (df['Close'].shift(1) <= df['SMA'].shift(1))
    sell_signals = (df['Close'] < df['SMA']) & (df['Close'].shift(1) >= df['SMA'].shift(1))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(df.index, df['Close'], label='Close Price', linewidth=1.2, color='steelblue')
    ax1.plot(df.index, df['SMA'], label='SMA 50', linestyle='--', linewidth=1.2, color='orange')

    ax1.fill_between(df.index, df['Close'], df['SMA'], where=df['Bullish'], color='green', alpha=0.15, label='Bullish Zone')
    ax1.fill_between(df.index, df['Close'], df['SMA'], where=df['Bearish'], color='red', alpha=0.15, label='Bearish Zone')

    ax1.plot(df.index[buy_signals], df['Close'][buy_signals], marker='^', color='darkgreen', label='Buy Signal', linestyle='None', markersize=6, alpha=0.9)
    ax1.plot(df.index[sell_signals], df['Close'][sell_signals], marker='v', color='darkred', label='Sell Signal', linestyle='None', markersize=6, alpha=0.9)

    ax1.set_ylabel("Prix")
    ax1.set_title(title)
    ax1.legend()
    ax1.grid(True)

    volume_colors = ['green' if df['Close'].iloc[i] > df['Close'].iloc[i-1] else 'red' for i in range(1, len(df))]
    volume_colors.insert(0, 'gray')  # première valeur

    ax2.bar(df.index, df['Volume'], width=1.0, color=volume_colors, alpha=0.6)
    ax2.set_ylabel("Volume")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_dual_pullbacks_comparison(df, title_left="Pullbacks SMA", title_right="Pullbacks Fibonacci"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6), sharey=True)

    ax1.plot(df.index, df['Close'], label='Close Price', color='steelblue', linewidth=1.2)
    ax1.plot(df.index, df['SMA'], label='SMA 50', linestyle='--', color='orange', linewidth=1.2)
    ax1.scatter(df.index[df['pullback_sma']], df['Close'][df['pullback_sma']], color='blue', marker='o', s=30, label='Pullback SMA')
    ax1.set_title(title_left)
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Prix")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(df.index, df['Close'], label='Close Price', color='steelblue', linewidth=1.2)
    ax2.plot(df.index, df['SMA'], label='SMA 50', linestyle='--', color='orange', linewidth=1.2)
    ax2.scatter(df.index[df['pullback_fib']], df['Close'][df['pullback_fib']], color='darkorange', marker='x', s=40, label='Pullback Fibonacci')
    ax2.set_title(title_right)
    ax2.set_xlabel("Date")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

def plot_minor_trend_signals(df, title="Minor Trend Reversals"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='steelblue')
    plt.plot(df.index, df['ma_20'], label='MA 20', linestyle='--', color='orange')

    plt.scatter(df.index[df['minor_trend_signal']], df['Close'][df['minor_trend_signal']],
                color='purple', marker='^', s=80, label='Minor Reversal Signal')

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_full_minor_trend_dashboard(df, title="Minor Trend Dashboard"):
    fig, axs = plt.subplots(4, 1, figsize=(16, 12), sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})

    axs[0].plot(df.index, df['Close'], label='Close', color='steelblue')
    axs[0].plot(df.index, df['ma_20'], label='MA 20', linestyle='--', color='orange')
    axs[0].plot(df.index, df['upper_bb'], label='Upper BB', linestyle='--', color='green', alpha=0.5)
    axs[0].plot(df.index, df['lower_bb'], label='Lower BB', linestyle='--', color='red', alpha=0.5)
    axs[0].scatter(df.index[df['micro_rebound']], df['Close'][df['micro_rebound']],
                   label='Micro Rebound', color='purple', marker='^', s=80)
    axs[0].set_title(title)
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(df.index, df['macd'], label='MACD', color='blue')
    axs[1].plot(df.index, df['macd_signal'], label='Signal', color='orange', linestyle='--')
    axs[1].fill_between(df.index, df['macd'] - df['macd_signal'], 0,
                        where=(df['macd'] > df['macd_signal']),
                        interpolate=True, color='green', alpha=0.3, label='Bullish MACD')
    axs[1].fill_between(df.index, df['macd'] - df['macd_signal'], 0,
                        where=(df['macd'] < df['macd_signal']),
                        interpolate=True, color='red', alpha=0.3, label='Bearish MACD')
    axs[1].set_ylabel("MACD")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(df.index, df['rsi'], label='RSI', color='darkblue')
    axs[2].axhline(70, linestyle='--', color='red', alpha=0.5)
    axs[2].axhline(30, linestyle='--', color='green', alpha=0.5)
    axs[2].fill_between(df.index, df['rsi'], 70, where=(df['rsi'] > 70), color='red', alpha=0.1)
    axs[2].fill_between(df.index, df['rsi'], 30, where=(df['rsi'] < 30), color='green', alpha=0.1)
    axs[2].set_ylabel("RSI")
    axs[2].legend()
    axs[2].grid(True)

    signal = df['minor_trend_signal'].astype(int)
    axs[3].bar(df.index, signal, color='purple', alpha=0.6, label='Signal')
    axs[3].set_ylabel("Signal")
    axs[3].set_yticks([0, 1])
    axs[3].legend()
    axs[3].grid(True)

    plt.tight_layout()
    plt.show()

def plot_fibonacci_retracement(df, lookback=20, title="Fibonacci Retracement (Last Swing) - Improved"):

    df = df.copy().tail(lookback)
    close = df['Close']
    high = close.max()
    low = close.min()
    swing_high_idx = close.idxmax()
    swing_low_idx = close.idxmin()
    uptrend = swing_high_idx > swing_low_idx

    if uptrend:
        levels = {
            '0.0': high,
            '0.382': high - 0.382 * (high - low),
            '0.5': high - 0.5 * (high - low),
            '0.618': high - 0.618 * (high - low),
            '1.0': low
        }
    else:
        levels = {
            '0.0': low,
            '0.382': low + 0.382 * (high - low),
            '0.5': low + 0.5 * (high - low),
            '0.618': low + 0.618 * (high - low),
            '1.0': high
        }

    plt.figure(figsize=(14, 6))
    plt.plot(df.index, close, label='Close Price', color='steelblue', linewidth=1.5)

    plt.scatter(swing_high_idx, high, color='green', marker='^', s=100, label='Swing High')
    plt.scatter(swing_low_idx, low, color='red', marker='v', s=100, label='Swing Low')

    for key, value in levels.items():
        plt.axhline(value, linestyle='--', alpha=0.6, color='grey')
        plt.text(df.index[-1], value, f'{key} ({value:.2f})', va='center', ha='left',
                 backgroundcolor='white', fontsize=9)

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Prix de clôture")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_signals(df: pd.DataFrame, trades: list, title="Buy/Sell/Short Signals with Zones"):
    df = df.copy().dropna()

    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(16, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    ax.plot(df.index, df['Close'], label='Cours', color='black', linewidth=1.5)
    ax.plot(df.index, df['SMA50'], label='SMA50', linestyle='--', color='royalblue', alpha=0.8)

    ax.scatter(df[df['buy_signal']].index, df[df['buy_signal']]['Close'],
               marker='^', color='green', label='Achat', s=90, zorder=3)
    ax.scatter(df[df['sell_signal']].index, df[df['sell_signal']]['Close'],
               marker='v', color='red', label='Vente', s=90, zorder=3)
    ax.scatter(df[df['short_signal']].index, df[df['short_signal']]['Close'],
               marker='x', color='darkorange', label='Short', s=90, linewidths=2, zorder=3)

    for trade in trades:
        entry = pd.to_datetime(trade['date_entry'])
        exit = pd.to_datetime(trade['date_exit'])
        if entry in df.index and exit in df.index:
            entry_price = df.loc[entry, 'Close']
            exit_price = df.loc[exit, 'Close']
            is_long = trade["type"] == "long"

            color_zone = 'lightgreen' if is_long else 'lightcoral'
            ax.axvspan(entry, exit, color=color_zone, alpha=0.15)

            color_line = 'forestgreen' if trade['pnl_$'] > 0 else 'firebrick'
            ax.plot([entry, exit], [entry_price, exit_price],
                    color=color_line, linestyle='-', linewidth=1.2, alpha=0.9)

    for t in trades:
        if t["pct_return"] < 0:
            exit_date = pd.to_datetime(t["date_exit"])
            if exit_date in df.index:
                price = df.loc[exit_date, 'Close']
                ax.scatter(exit_date, price, marker='X', color='black', label='Stop Loss', s=80, zorder=4)

    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    legend_cleaned = [(h, l) for h, l in zip(handles, labels) if l not in seen and not seen.add(l)]
    ax.legend(*zip(*legend_cleaned), loc='lower left', fontsize=10)

    ax.set_title(title, fontsize=15, fontweight='bold')
    ax.set_xlabel("Date")
    ax.set_ylabel("Prix (€)")
    ax.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_simulation_distribution(returns, best=None, worst=None, avg=None):
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")

    sns.histplot(returns, bins=20, kde=False, color="#5DADE2", edgecolor='white', alpha=0.9)

    if avg is not None:
        plt.axvline(avg, color='crimson', linestyle='--', linewidth=2)
        plt.text(avg, plt.ylim()[1]*0.92, f"Moyenne: {avg:.2f}%",
                 color='crimson', ha='center', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='crimson', boxstyle='round,pad=0.4'))

    if best:
        plt.axvline(best['total_return'], color='seagreen', linestyle='--', linewidth=2)
        plt.text(best['total_return'], plt.ylim()[1]*0.82, f"Max: {best['total_return']:.2f}%",
                 color='seagreen', ha='center', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='seagreen', boxstyle='round,pad=0.4'))

    if worst:
        plt.axvline(worst['total_return'], color='darkorange', linestyle='--', linewidth=2)
        plt.text(worst['total_return'], plt.ylim()[1]*0.72, f"Min: {worst['total_return']:.2f}%",
                 color='darkorange', ha='center', fontsize=10,
                 bbox=dict(facecolor='white', edgecolor='darkorange', boxstyle='round,pad=0.4'))

    plt.title("Distribution des rendements simulés de la stratégie", fontsize=16, fontweight='bold', pad=15)
    plt.xlabel("Rendement (%)", fontsize=12)
    plt.ylabel("Fréquence", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.show()


def plot_heatmap_returns(sim_results):
    df = pd.DataFrame(sim_results)
    pivot = df.pivot_table(index="mu", columns="sigma", values="total_return")

    plt.figure(figsize=(10, 7))
    ax = sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="rocket_r",
        cbar_kws={
            "label": "Rendement (%)",
            "shrink": 0.8,
            "orientation": "vertical"
        },
        linewidths=0.5,
        linecolor='white',
        square=True
    )

    ax.set_title("Rendement de la stratégie\nselon μ (drift) et σ (volatilité)",
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("σ (volatilité)", fontsize=12)
    ax.set_ylabel("μ (drift)", fontsize=12)

    ax.tick_params(axis='both', labelsize=10)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.show()


def plot_simulated_portfolios(simulations, title="Évolution des portefeuilles simulés"):
    best_sim = max(simulations, key=lambda x: x['total_return'])
    worst_sim = min(simulations, key=lambda x: x['total_return'])

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    ax.set_facecolor("#f9f9f9")

    rect = patches.Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes,
        facecolor='none', edgecolor='gray', linewidth=1.5,
        linestyle='--'
    )
    ax.add_patch(rect)

    for sim in simulations:
        returns = sim['returns']
        if not isinstance(returns, pd.Series):
            returns = pd.Series(returns)

        if sim not in [best_sim, worst_sim]:
            plt.plot(returns.index, returns.values, color='lightsteelblue', alpha=0.2, linewidth=0.8)


    returns = worst_sim['returns']
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    plt.plot(returns.index, returns.values, color='firebrick', linewidth=2.5,
             label=f"Pire: μ={worst_sim['mu']:.4f}, σ={worst_sim['sigma']:.4f}, {worst_sim['total_return']:.2f}%")

    returns = best_sim['returns']
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    plt.plot(returns.index, returns.values, color='seagreen', linewidth=2.5,
             label=f"Meilleure: μ={best_sim['mu']:.4f}, σ={best_sim['sigma']:.4f}, {best_sim['total_return']:.2f}%")

    plt.title(title, fontsize=16, fontweight='bold', loc='center')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Valeur du portefeuille", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc='upper left', fontsize=10, frameon=True, facecolor='white', edgecolor='gray')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.tight_layout()
    plt.box(True)
    plt.show()