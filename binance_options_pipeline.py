"""
Pipeline de datos — Binance Options (European Options API)
=============================================================

Descarga en tiempo real:
  1. Catálogo de contratos vigentes (exchangeInfo)
  2. Mark price, IV (bid/ask/mark) y griegas por contrato (mark)
  3. Volumen y estadísticas 24h (ticker)

Combina todo en un DataFrame por "cadena de opciones" (options chain) y
construye la superficie de volatilidad implícita (strike x vencimiento)
por subyacente.

Requisitos:
    pip install requests pandas numpy --break-system-packages

Nota: son endpoints PÚBLICOS (market data), no requieren API key.
Base URL oficial: https://eapi.binance.com
"""

from __future__ import annotations

import time
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("binance_options")

BASE_URL = "https://eapi.binance.com"
ENDPOINTS = {
    "exchange_info": "/eapi/v1/exchangeInfo",
    "mark": "/eapi/v1/mark",       # markPrice, bidIV, askIV, markIV, delta, gamma, theta, vega, riskFreeInterest
    "ticker": "/eapi/v1/ticker",   # volumen, high/low 24h, priceChangePercent
    "index": "/eapi/v1/index",     # precio índice del subyacente (spot real usado para valuar las opciones)
}

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Cliente HTTP mínimo con reintentos
# ---------------------------------------------------------------------------
def _get(path: str, params: dict | None = None, retries: int = 3, backoff: float = 1.5) -> dict | list:
    url = f"{BASE_URL}{path}"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            log.warning("Intento %d/%d falló para %s: %s", attempt, retries, path, exc)
            if attempt == retries:
                raise
            time.sleep(backoff * attempt)


# ---------------------------------------------------------------------------
# Parsing del símbolo: "BTC-260925-70000-C"  ->  underlying, expiry, strike, side
# ---------------------------------------------------------------------------
def parse_symbol(symbol: str) -> dict:
    underlying, expiry_raw, strike_raw, side_raw = symbol.split("-")
    expiry = datetime.strptime(expiry_raw, "%y%m%d").replace(tzinfo=timezone.utc)
    return {
        "underlying": underlying,
        "expiry": expiry,
        "strike": float(strike_raw),
        "option_type": "CALL" if side_raw == "C" else "PUT",
    }


# ---------------------------------------------------------------------------
# Descarga de datos crudos
# ---------------------------------------------------------------------------
def fetch_exchange_info() -> pd.DataFrame:
    """Catálogo de contratos vigentes."""
    data = _get(ENDPOINTS["exchange_info"])
    contracts = data.get("optionSymbols", data.get("symbols", []))
    df = pd.DataFrame(contracts)
    log.info("exchangeInfo: %d contratos vigentes", len(df))
    return df


def fetch_mark_prices(symbol: str | None = None) -> pd.DataFrame:
    """Mark price + IV (bid/ask/mark) + griegas. Si symbol=None, trae todos."""
    params = {"symbol": symbol} if symbol else None
    data = _get(ENDPOINTS["mark"], params=params)
    df = pd.DataFrame(data)
    numeric_cols = ["markPrice", "bidIV", "askIV", "markIV", "delta",
                     "theta", "gamma", "vega", "highPriceLimit", "lowPriceLimit", "riskFreeInterest"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_ticker(symbol: str | None = None) -> pd.DataFrame:
    """Volumen y estadísticas de 24h."""
    params = {"symbol": symbol} if symbol else None
    data = _get(ENDPOINTS["ticker"], params=params)
    df = pd.DataFrame(data)
    for col in ["volume", "amount", "lastPrice", "priceChangePercent", "openPrice", "highPrice", "lowPrice"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fetch_index_price(underlying: str) -> float:
    """
    Precio índice del subyacente (ej. 'BTCUSDT') que Binance usa como spot
    real para valuar las opciones -- lo necesitas para tu propio Black-Scholes.
    """
    data = _get(ENDPOINTS["index"], params={"underlying": f"{underlying}USDT"})
    return float(data["indexPrice"])


# ---------------------------------------------------------------------------
# Construcción de la cadena de opciones combinada
# ---------------------------------------------------------------------------
@dataclass
class OptionsChain:
    underlying: str
    chain: pd.DataFrame          # una fila por contrato, con IV/griegas/volumen
    fetched_at: datetime


def build_options_chain(underlying: str | None = None) -> OptionsChain:
    """
    Combina mark (IV + griegas) con ticker (volumen) y descompone el símbolo
    en underlying / expiry / strike / tipo. Si underlying se especifica
    (ej. 'BTC'), filtra solo esos contratos.
    """
    mark_df = fetch_mark_prices()
    ticker_df = fetch_ticker()

    merged = mark_df.merge(
        ticker_df[["symbol", "volume", "amount", "lastPrice", "priceChangePercent"]],
        on="symbol", how="left",
    )

    parsed = merged["symbol"].apply(parse_symbol).apply(pd.Series)
    merged = pd.concat([merged, parsed], axis=1)

    now = datetime.now(timezone.utc)
    merged["days_to_expiry"] = (merged["expiry"] - now).dt.total_seconds() / 86400
    merged["T_years"] = merged["days_to_expiry"] / 365

    if underlying:
        merged = merged[merged["underlying"] == underlying.upper()].copy()

    merged = merged.sort_values(["underlying", "expiry", "strike", "option_type"]).reset_index(drop=True)

    cols = ["symbol", "underlying", "expiry", "days_to_expiry", "T_years", "strike", "option_type",
            "markPrice", "markIV", "bidIV", "askIV", "delta", "gamma", "theta", "vega",
            "volume", "lastPrice", "riskFreeInterest"]
    cols = [c for c in cols if c in merged.columns]

    log.info("Cadena construida: %d contratos%s", len(merged),
              f" para {underlying}" if underlying else "")

    return OptionsChain(underlying=underlying or "ALL", chain=merged[cols], fetched_at=now)


# ---------------------------------------------------------------------------
# Superficie de volatilidad implícita
# ---------------------------------------------------------------------------
def build_vol_surface(chain: pd.DataFrame, option_type: str = "CALL") -> pd.DataFrame:
    """
    Pivotea la cadena en una matriz strike (filas) x vencimiento (columnas)
    de IV implícita (markIV). Útil para visualizar smile/skew por vencimiento.
    """
    subset = chain[chain["option_type"] == option_type]
    surface = subset.pivot_table(index="strike", columns="days_to_expiry", values="markIV")
    return surface.sort_index()


# ---------------------------------------------------------------------------
# Filtro de liquidez y visualización del smile
# ---------------------------------------------------------------------------
def filter_liquid(chain: pd.DataFrame, min_volume: float = 0.0, min_last_price: float = 0.0) -> pd.DataFrame:
    """
    Filtra contratos sin volumen/actividad real. Por defecto no filtra nada
    (min_volume=0); súbelo (ej. min_volume=1) para quedarte solo con strikes
    que de verdad tradean, evitando ruido de contratos ilíquidos en tu smile.
    """
    mask = (chain["volume"].fillna(0) >= min_volume) & (chain["lastPrice"].fillna(0) >= min_last_price)
    return chain[mask].copy()


def plot_smile(chain: pd.DataFrame, expiry_days: float, option_type: str = "CALL", tolerance: float = 0.05):
    """
    Grafica IV vs strike para un vencimiento específico (± tolerance en días
    para agarrar el vencimiento exacto sin errores de redondeo de tiempo).
    Requiere matplotlib: pip install matplotlib --break-system-packages
    """
    import matplotlib.pyplot as plt

    subset = chain[
        (chain["option_type"] == option_type)
        & (chain["days_to_expiry"].between(expiry_days - tolerance, expiry_days + tolerance))
    ].sort_values("strike")

    if subset.empty:
        log.warning("No hay contratos %s cerca de %.2f días a vencimiento", option_type, expiry_days)
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(subset["strike"], subset["markIV"] * 100, marker="o", label="markIV")
    ax.plot(subset["strike"], subset["bidIV"] * 100, linestyle="--", alpha=0.5, label="bidIV")
    ax.plot(subset["strike"], subset["askIV"] * 100, linestyle="--", alpha=0.5, label="askIV")
    ax.set_xlabel("Strike")
    ax.set_ylabel("Volatilidad implícita (%)")
    ax.set_title(f"Smile de IV — {chain['underlying'].iloc[0]} {option_type} — {expiry_days:.1f} días a vencimiento")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = OUTPUT_DIR / f"smile_{chain['underlying'].iloc[0]}_{option_type}_{expiry_days:.0f}d.png"
    fig.savefig(out_path, dpi=150)
    log.info("Gráfica guardada: %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def save_snapshot(chain_obj: OptionsChain) -> Path:
    ts = chain_obj.fetched_at.strftime("%Y%m%d_%H%M%S")
    fname = OUTPUT_DIR / f"options_chain_{chain_obj.underlying}_{ts}.csv"
    chain_obj.chain.to_csv(fname, index=False)
    log.info("Guardado: %s", fname)
    return fname


# ---------------------------------------------------------------------------
# Ejecución de ejemplo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Trae la cadena completa de BTC
    btc_chain = build_options_chain(underlying="BTC")
    print(btc_chain.chain.head(10))

    # 2. Guarda snapshot en disco (útil para acumular historia con un cron job)
    save_snapshot(btc_chain)

    # 3. Superficie de volatilidad (calls)
    surface = build_vol_surface(btc_chain.chain, option_type="CALL")
    print("\nSuperficie de IV (CALL) — filas=strike, columnas=días a vencimiento:")
    print(surface.round(3))

    # 4. Ejemplo: detectar el vencimiento más cercano y su smile
    if not btc_chain.chain.empty:
        nearest_expiry_days = btc_chain.chain["days_to_expiry"].min()
        near = btc_chain.chain[
            (btc_chain.chain["days_to_expiry"] == nearest_expiry_days)
            & (btc_chain.chain["option_type"] == "CALL")
        ]
        print(f"\nSmile del vencimiento más próximo (~{nearest_expiry_days:.1f} días):")
        print(near[["strike", "markIV", "delta", "volume"]])

        # 5. Filtra solo contratos con actividad real y grafica el smile
        liquid_chain = filter_liquid(btc_chain.chain, min_volume=1.0)
        plot_smile(liquid_chain, expiry_days=nearest_expiry_days, option_type="CALL")
