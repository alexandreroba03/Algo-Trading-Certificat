from src.Data import DataFactory
import pandas as pd
import ta

class TechInd:
    def __init__(self, ticker=None, start_date=None, end_date=None, df=None, window_bb=20, window_rsi=14):
        self.window_bb = window_bb
        self.window_rsi = window_rsi

        if df is not None:
            self.df = df.copy()
        elif ticker and start_date and end_date:
            self.df = DataFactory(ticker, start_date, end_date).get_closing_price()
        else:
            raise ValueError("Veuillez fournir soit un DataFrame `df`, soit un `ticker` + `start_date` + `end_date`.")

    def add_moving_average(self, df: pd.DataFrame) -> pd.DataFrame:
        df[f'ma_{self.window_bb}'] = df['Close'].rolling(window=self.window_bb).mean()
        return df

    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        ma = df['Close'].rolling(window=self.window_bb).mean()
        std = df['Close'].rolling(window=self.window_bb).std()
        df['upper_bb'] = ma + 2 * std
        df['lower_bb'] = ma - 2 * std
        return df

    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        df['rsi'] = ta.momentum.RSIIndicator(close=df['Close'], window=self.window_rsi).rsi()
        return df

    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        macd = ta.trend.MACD(close=df['Close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        return df

    def compute_all(self) -> pd.DataFrame:
        df = self.df.copy()
        df = self.add_moving_average(df)
        df = self.add_bollinger_bands(df)
        df = self.add_rsi(df)
        df = self.add_macd(df)
        return df