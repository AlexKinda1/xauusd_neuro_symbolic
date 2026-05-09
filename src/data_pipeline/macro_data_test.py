import yfinance
import os
import pandas as pd


def load_macro_data(start_date: str, end_date: str):
    """
    Télécharge les données macroéconomiques pour un ticker donné entre les dates spécifiées.
    """
    tickers = ["^VIX", "DX-Y.NYB", "^TNX"]   
    
    #folder_path = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1\csv"   
    
    df_macro = yfinance.download(tickers, start=start_date, end=end_date, progress=False)
    
    df_macro.rename(columns={
        '^VIX': 'VIX_Close',
        'DX-Y.NYB': 'DXY_Close',
        '^TNX': 'US10Y_Close'
    }, inplace=True)
    
    print(df_macro.info(f"Macro data downloaded for tickers: {tickers} from {start_date} to {end_date}"))
    
    return df_macro


def load_micro_data(file_path: str):
    
    H1_data = os.path.join(file_path, "XAUUSD_H1_Features_Final.csv")
    
    micro_data = pd.read_csv(H1_data)
    
    return micro_data


# Exemple d'utilisation
if __name__ == "__main__":
    start_date = "2009-03-17"
    end_date = "2026-04-10"
    
    #file_path = r"C:\Users\dell\Desktop\xauusd_neuro_symbolic\XAUUSD_data_M1"
    
    #df_micro = load_micro_data(file_path)
    
    df_macro = load_macro_data(start_date, end_date)
    df_macro.to_csv("macro_data.csv", index=True)
    
    print(df_macro.head())
    # print(df_micro.info())
    # print(df_micro.describe())
    # print(df_micro.head())