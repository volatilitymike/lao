# pages/components/sectorClassifier.py

def assign_sector(ticker: str) -> str:
    """
    Simple sector mapper for JSON export.
    Input ticker must already be lowercase.
    """

    ticker = ticker.lower()

    groups = {
        "ETFs": ["spy", "qqq"],
        "finance": ["wfc", "c", "jpm", "bac", "hood", "coin", "pypl"],
        "Semiconductors": ["nvda", "avgo", "amd", "mu", "mrvl", "qcom", "smci"],
        "Software": ["msft", "pltr", "aapl", "googl", "meta", "uber", "tsla", "amzn"],
        "Futures": ["nq", "es", "gc", "ym", "cl"],
    }

    for sector, tickers in groups.items():
        if ticker in tickers:
            return sector

    return "Other"
