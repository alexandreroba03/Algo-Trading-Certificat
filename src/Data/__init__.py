import yfinance as yf
import pandas as pd

class DataFactory:
    def __init__(self, ticker: str, start_date: str, end_date: str):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def get_closing_price(self):
        df = yf.download(
            tickers=self.ticker,
            start=self.start_date,
            end=self.end_date,
            auto_adjust=True
        )

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        print(f"\nDonnées téléchargées pour {self.ticker} :\n{df.tail()}")
        print(f"Colonnes présentes : {df.columns.tolist()}")

        expected_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in expected_cols:
            if col not in df.columns:
                raise ValueError(f"Colonne manquante dans les données téléchargées : '{col}'")

        return df[expected_cols].copy()