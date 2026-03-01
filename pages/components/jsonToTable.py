import pandas as pd

def flatten_json(y, prefix=""):
    """Flatten nested dicts/lists into dot.notation keys."""
    out = {}

    def _flatten(x, name=""):
        if isinstance(x, dict):
            for k, v in x.items():
                _flatten(v, f"{name}{k}.")
        elif isinstance(x, list):
            for i, v in enumerate(x):
                _flatten(v, f"{name}{i}.")
        else:
            out[name[:-1]] = x   # remove last dot

    _flatten(y)
    return out


def json_to_dataframe(json_payload: dict) -> pd.DataFrame:
    """
    Converts a full JSON payload into a single-row DataFrame
    with flattened columns.
    """
    flat = flatten_json(json_payload)
    df = pd.DataFrame([flat])
    return df
