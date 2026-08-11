"""
Captura de un solo snapshot -- pensado para correr en GitHub Actions
=============================================================================

A diferencia de run_loop.py (que corre en tu compu con un while infinito),
este script hace UNA sola captura y termina. La "repetición cada 15 minutos"
la maneja GitHub Actions con un cron, no este script.

Reutiliza la misma lógica de acumulación de run_loop.py (un CSV por día,
agregando filas con su timestamp).

Uso:
    python collect_snapshot.py BTC
"""

from __future__ import annotations

import sys
import logging

from run_loop import run_once, history_file_for_today

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("collect_snapshot")

if __name__ == "__main__":
    underlying = sys.argv[1] if len(sys.argv) > 1 else "BTC"
    try:
        n_rows = run_once(underlying)
        log.info("Snapshot capturado: %d contratos agregados a %s",
                  n_rows, history_file_for_today(underlying).name)
    except Exception as exc:
        log.error("La captura falló: %s", exc)
        sys.exit(1)  # código de error distinto de 0 -> GitHub Actions marca el run como fallido, útil para notificarte
