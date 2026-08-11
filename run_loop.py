"""
Loop de recolección de historia -- Binance Options
=============================================================================

Corre el pipeline de datos cada X minutos y va acumulando cada snapshot
(con su timestamp) en un solo CSV por día -- en vez de crear un archivo
nuevo por corrida, todo se va agregando a un archivo creciente, que es
mucho más útil para análisis después (ej. forecasting de volatilidad
realizada, o medir si la "prima de salto" del corto plazo es persistente).

Diseño simple a propósito, pensado para dejarlo corriendo en una terminal:
  - Si una corrida falla (ej. se cae internet un momento), NO se detiene
    todo el loop -- se registra el error y se reintenta en la siguiente
    vuelta.
  - Ctrl+C lo detiene de forma segura en cualquier momento.
  - Cada snapshot queda con su propia columna "snapshot_time" para poder
    reconstruir la serie de tiempo después.

Uso:
    python run_loop.py BTC --interval 15
    python run_loop.py BTC --interval 15 --max-runs 100   (para pruebas cortas)

Deja la terminal abierta mientras quieras seguir recolectando datos.
"""

from __future__ import annotations

import argparse
import time
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from binance_options_pipeline import build_options_chain, fetch_index_price, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("options_history_loop")

HISTORY_DIR = OUTPUT_DIR / "history"
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def history_file_for_today(underlying: str) -> Path:
    """Un archivo por día -- evita que un solo CSV crezca sin límite para siempre."""
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    return HISTORY_DIR / f"history_{underlying}_{today}.csv"


def run_once(underlying: str) -> int:
    """Ejecuta una captura y la agrega al CSV del día. Devuelve cuántas filas agregó."""
    chain_obj = build_options_chain(underlying=underlying)
    spot = fetch_index_price(underlying)

    snapshot_time = chain_obj.fetched_at
    df = chain_obj.chain.copy()
    df.insert(0, "snapshot_time", snapshot_time)
    df.insert(1, "spot_index", spot)

    out_path = history_file_for_today(underlying)
    file_exists = out_path.exists()
    df.to_csv(out_path, mode="a", header=not file_exists, index=False)

    return len(df)


def main():
    parser = argparse.ArgumentParser(description="Loop de recolección de historia de opciones de Binance")
    parser.add_argument("underlying", nargs="?", default="BTC", help="Subyacente (ej. BTC, ETH). Default: BTC")
    parser.add_argument("--interval", type=int, default=15, help="Minutos entre capturas. Default: 15")
    parser.add_argument("--max-runs", type=int, default=None, help="Número máximo de corridas (default: infinito)")
    args = parser.parse_args()

    log.info("Iniciando loop para %s cada %d minutos. Ctrl+C para detener.", args.underlying, args.interval)
    log.info("Historia se guarda en: %s", HISTORY_DIR)

    run_count = 0
    try:
        while True:
            run_count += 1
            if args.max_runs and run_count > args.max_runs:
                log.info("Se alcanzó el máximo de %d corridas. Terminando.", args.max_runs)
                break

            try:
                n_rows = run_once(args.underlying)
                log.info("Corrida #%d: %d contratos agregados a %s",
                          run_count, n_rows, history_file_for_today(args.underlying).name)
            except Exception as exc:
                # No dejamos que un error puntual (ej. caída de red, timeout de la API)
                # tumbe el loop completo -- se registra y se sigue en la próxima vuelta.
                log.error("Corrida #%d falló: %s. Reintentando en la siguiente vuelta.", run_count, exc)

            log.info("Esperando %d minutos hasta la siguiente captura...", args.interval)
            time.sleep(args.interval * 60)

    except KeyboardInterrupt:
        log.info("Detenido manualmente (Ctrl+C) después de %d corridas.", run_count)


if __name__ == "__main__":
    main()
