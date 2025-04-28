import pandas as pd
import json
import plotly.graph_objects as go
import os

class Report:
    def __init__(self, df: pd.DataFrame, result: dict):
        self.df = df.copy()
        self.result = result

    def SummaryTable(self):
        perf = self.result.get('performance', self.result.get('strategy_summary', {}))
        summary = pd.DataFrame({
            'Metric': [
                'Starting Balance', 'Ending Balance', 'Total Return (%)',
                'Average Trade Return (%)', 'Win Ratio (%)',
                'Number of Trades', 'Number of Short Trades', 'Max Drawdown (%)',
                'Stop Loss Triggers', 'Total Short Candidates', 'Short Signals Detected', 'Short Signals Ignored'
            ],
            'Value': [
                perf.get('starting_balance', 'N/A'),
                perf.get('ending_balance', 'N/A'),
                perf.get('total_return_%', 'N/A'),
                perf.get('average_trade_return_%', 'N/A'),
                perf.get('win_ratio_%', 'N/A'),
                perf.get('number_of_trades', 'N/A'),
                perf.get('number_of_short_trades', 'N/A'),
                perf.get('max_drawdown_%', 'N/A'),
                perf.get('stop_loss_triggers', 'N/A'),
                perf.get('short_candidates_total', 'N/A'),
                perf.get('short_signals_detected', 'N/A'),
                perf.get('short_signals_ignored', 'N/A')
            ]
        })
        return summary

    def ExtendedStatsTable(self):
        trades = pd.DataFrame(self.result.get("trades", []))
        if trades.empty:
            return pd.DataFrame()

        long_trades = trades[trades["type"] == "long"]
        short_trades = trades[trades["type"] == "short"]

        stats = {
            "🟢 Long Trades Count": len(long_trades),
            "🔴 Short Trades Count": len(short_trades),
            "🟢 Average Long PnL ($)": round(long_trades['pnl_$'].mean(), 2) if not long_trades.empty else 0,
            "🔴 Average Short PnL ($)": round(short_trades['pnl_$'].mean(), 2) if not short_trades.empty else 0,
            "🔴 Average Short Return (%)": round(short_trades['pct_return'].mean(), 2) if not short_trades.empty else 0,
            "🔴 Max Short Duration (days)": short_trades['duration_days'].max() if not short_trades.empty else 0
        }

        return pd.DataFrame(stats.items(), columns=["Detailed Trade Stats", "Value"])

    def SignalStatsTable(self):
        signals_df = pd.DataFrame({
            'Buy Signals': self.df['buy_signal'].sum() if 'buy_signal' in self.df.columns else 0,
            'Sell Signals': self.df['sell_signal'].sum() if 'sell_signal' in self.df.columns else 0,
            'Short Signals': self.df['short_signal'].sum() if 'short_signal' in self.df.columns else 0
        }, index=[0])
        return signals_df.T.reset_index().rename(columns={'index': 'Signal Type', 0: 'Count'})

    def ExcelReport(self, file_path="Reporting/report.xlsx"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        trades_df = pd.DataFrame(self.result.get("trades", []))
        summary_df = self.SummaryTable()
        extended_df = self.ExtendedStatsTable()
        signal_stats_df = self.SignalStatsTable()

        signal_cols = [col for col in ['Close', 'buy_signal', 'sell_signal', 'short_signal']
                       if col in self.df.columns]
        signals_df = self.df[signal_cols].copy()
        if 'buy_signal' in signals_df.columns or 'sell_signal' in signals_df.columns:
            signals_df = signals_df[(signals_df.get('buy_signal', False)) |
                                    (signals_df.get('sell_signal', False)) |
                                    (signals_df.get('short_signal', False))]

        with pd.ExcelWriter(file_path) as writer:
            summary_df.to_excel(writer, sheet_name='Strategy Summary', index=False)
            extended_df.to_excel(writer, sheet_name='Detailed Trade Stats', index=False)
            signal_stats_df.to_excel(writer, sheet_name='Signal Counts', index=False)
            trades_df.to_excel(writer, sheet_name='Trades', index=False)
            if not signals_df.empty:
                signals_df.to_excel(writer, sheet_name='Signals', index=True)

    def JsonReport(self, file_path="Reporting/result.json"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        self.result['buydates'] = [str(i.date()) for i in self.df[self.df.get('buy_signal', False)].index]
        self.result['selldates'] = [str(i.date()) for i in self.df[self.df.get('sell_signal', False)].index]

        if 'pullback_sma' in self.df.columns:
            self.result['pullbacks_sma'] = [str(d.date()) for d in self.df[self.df['pullback_sma']].index]
        if 'pullback_fib' in self.df.columns:
            self.result['pullbacks_fibonacci'] = [str(d.date()) for d in self.df[self.df['pullback_fib']].index]

        self.result['strategy_summary'] = self.result.get('performance', {})
        self.result.pop('performance', None)

        with open(file_path, "w") as f:
            json.dump(self.result, f, indent=4)

    def print_summary(self):
        summary = self.result.get('strategy_summary', self.result.get('performance', {}))
        trades_df = pd.DataFrame(self.result.get("trades", []))
        signal_counts = self.SignalStatsTable()

        print("\n Strategy Summary")
        print("-" * 40)
        for key, val in summary.items():
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                print(f" {label:<30}: {val:.2f}")
            else:
                print(f" {label:<30}: {val}")
        print("-" * 40)

        if not trades_df.empty:
            long_trades = trades_df[trades_df["type"] == "long"]
            short_trades = trades_df[trades_df["type"] == "short"]
            print(" Trade Stats")
            print(f"🟢 Long trades           : {len(long_trades)}")
            print(f"🔴 Short trades          : {len(short_trades)}")
            if not long_trades.empty:
                avg_long = round(long_trades["pnl_$"].mean(), 2)
                print(f" PnL moyen sur longs     : {avg_long} $")
            if not short_trades.empty:
                avg_short = round(short_trades["pnl_$"].mean(), 2)
                ret_short = round(short_trades["pct_return"].mean(), 2)
                max_dur_short = short_trades["duration_days"].max()
                print(f" PnL moyen sur shorts    : {avg_short} $")
                print(f" Rendement moyen shorts  : {ret_short} %")
                print(f" Durée max short (jours) : {max_dur_short}")
            print("-" * 40)

        print(" Signals Summary")
        for _, row in signal_counts.iterrows():
            print(f"{row['Signal Type']:<25}: {row['Count']}")
        print("-" * 40)

    def InteractiveChart(self, file_path="Reporting/signals_chart.html"):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        fig = go.Figure()

        if 'Close' in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df.index, y=self.df['Close'],
                mode='lines', name='Close Price',
                line=dict(color='black')
            ))

        if 'buy_signal' in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df[self.df['buy_signal']].index,
                y=self.df[self.df['buy_signal']]['Close'],
                mode='markers', name='Buy',
                marker=dict(color='green', symbol='triangle-up', size=10)
            ))

        if 'sell_signal' in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df[self.df['sell_signal']].index,
                y=self.df[self.df['sell_signal']]['Close'],
                mode='markers', name='Sell',
                marker=dict(color='red', symbol='triangle-down', size=10)
            ))

        if 'short_signal' in self.df.columns:
            fig.add_trace(go.Scatter(
                x=self.df[self.df['short_signal']].index,
                y=self.df[self.df['short_signal']]['Close'],
                mode='markers', name='Short',
                marker=dict(color='orange', symbol='x', size=9)
            ))

        fig.update_layout(
            title=" Trading Signals Overview",
            xaxis_title="Date",
            yaxis_title="Price",
            template="plotly_white"
        )
        fig.write_html(file_path)