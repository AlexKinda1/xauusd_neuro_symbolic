import yfinance

def load_macro_data(start_date: str, end_date: str):
    """
    Télécharge les données macroéconomiques pour un ticker donné entre les dates spécifiées.
    """
    tickers = ["^VIX", "DX-Y.NYB", "^TNX"]
    
    df_macro = yfinance.download(tickers, start=start_date, end=end_date, progress=False)
    
    print(df_macro.info(f"Macro data downloaded for tickers: {tickers} from {start_date} to {end_date}"))
    
    return df_macro

# Exemple d'utilisation
if __name__ == "__main__":
    start_date = "2020-01-01"
    end_date = "2024-01-01"
    
    df_macro = load_macro_data(start_date, end_date)
    df_macro.to_csv("macro_data.csv", index=True)
    print(df_macro.head())