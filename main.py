import os
import numpy as np
from src.Data import DataFactory
from src.Strategy import (
    analyze_stock_trends,
    detect_sma_pullbacks,
    detect_fibonacci_pullbacks,
    analyze_minor_trend,
    generate_signals,
    simple_backtest,
    simulate_price_series,
    build_synthetic_dataframe,
    run_multiple_simulations
)
from src.Reporting import Report
from src.Plot import (
    plot_dow_trend,
    plot_dual_pullbacks_comparison,
    plot_minor_trend_signals,
    plot_full_minor_trend_dashboard,
    plot_fibonacci_retracement,
    plot_signals,
    plot_simulation_distribution,
    plot_heatmap_returns,
    plot_simulated_portfolios
)
from src.TI import TechInd


# Paramètres
ticker = ("NVDA")
benchmark = "^GSPC"
start_date = "2020-01-01"
end_date = "2024-12-31"
initial_balance = 10000
show_graphs = True
export_reports = True

stock = DataFactory(ticker, start_date, end_date).get_closing_price()
index = DataFactory(benchmark, start_date, end_date).get_closing_price()

# Dow Theory
print("\n [DOW THEORY RESULT]")
dow_result = analyze_stock_trends(stock, index)
print(dow_result)

# Indicateurs techniques
ti = TechInd(ticker, start_date, end_date)
df_ti = ti.compute_all()

# Signaux de trading
final_df = generate_signals(
    df=df_ti,
    index_df=index,
    sma_window=50,
    fib_window=100,
    distance_sma_thresh=0.02,
    fib_tolerance=0.02,
    rsi_buy_thresh=35,
    rsi_sell_thresh=65,
    min_short_term_signals=2,
    min_short_signals=1,
    volume_lookback=20,
    volume_threshold=1.2
)

# Backtest
results = simple_backtest(
    final_df,
    initial_balance=initial_balance,
    stop_loss_long=-0.05,
    take_profit_long=0.10,
    stop_loss_short=-0.02,
    take_profit_short=0.06,
    trailing_trigger_long=0.05,
    trailing_stop_long=0.02,
    trailing_trigger_short=0.03,
    trailing_stop_short=0.015,
    cooldown_period=5
)

# Rapport et résumé console
trades = results["trades"]
summary = results['performance']

result = {
    "buydates": final_df[final_df['buy_signal']].index.tolist(),
    "selldates": final_df[final_df['sell_signal']].index.tolist(),
    "stoploss_dates": [],
    "trades": trades,
    "performance": summary
}

report = Report(final_df, result)
os.makedirs("Reporting", exist_ok=True)
if export_reports:
    report.ExcelReport("Reporting/report.xlsx")
    report.JsonReport("Reporting/result.json")
    report.InteractiveChart("Reporting/signals_chart.html")

report.print_summary()

if show_graphs:
    plot_signals(final_df, trades, title=f" Buy/Sell/Short Signals - {ticker}")
    df_minor = analyze_minor_trend(df_ti)
    plot_fibonacci_retracement(df_minor, lookback=20)
    plot_full_minor_trend_dashboard(df_minor)
    plot_minor_trend_signals(df_minor)
    stock['SMA'] = stock['Close'].rolling(window=50).mean()
    stock['pullback_sma'] = detect_sma_pullbacks(stock)
    stock['pullback_fib'] = detect_fibonacci_pullbacks(stock)
    plot_dual_pullbacks_comparison(stock)
    plot_dow_trend(stock, title=f" Dow Theory - {ticker}")

# Test de l'algorithme sur simulation de GBM
print("\n Simulated Market Test")

sim_results = run_multiple_simulations(n_simulations=10)
returns = [res["total_return"] for res in sim_results]

best_sim = max(sim_results, key=lambda x: x["total_return"])
worst_sim = min(sim_results, key=lambda x: x["total_return"])
avg_return = np.mean(returns)

print(f"\n→ Moyenne rendement simulé : {avg_return:.2f}%")
print(f"→ Meilleure simulation : mu={best_sim['mu']:.4f}, sigma={best_sim['sigma']:.4f}, rendement={best_sim['total_return']:.2f}%")
print(f"→ Pire simulation : mu={worst_sim['mu']:.4f}, sigma={worst_sim['sigma']:.4f}, rendement={worst_sim['total_return']:.2f}%")


plot_simulation_distribution(returns, best=best_sim, worst=worst_sim, avg=avg_return)
plot_heatmap_returns(sim_results)
plot_simulated_portfolios(sim_results)