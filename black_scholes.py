"""
Black-Scholes propio: precio, griegas e inversión de volatilidad implícita
=============================================================================

Implementación desde cero (sin librerías de opciones de terceros) para que
puedas auditar cada fórmula y compararla contra lo que Binance reporta en
/eapi/v1/mark (markIV, delta, gamma, theta, vega).

Convenciones:
    S       = precio spot del subyacente
    K       = strike
    T       = tiempo a vencimiento en AÑOS (no días)
    r       = tasa libre de riesgo anualizada, continua (Binance la da en
              riskFreeInterest dentro de /eapi/v1/mark)
    sigma   = volatilidad anualizada (ej. 0.55 = 55%)
    option_type = "CALL" o "PUT"

Theta se devuelve en la MISMA convención que Binance: valor absoluto por
día (Binance reporta theta como número positivo grande, ej. 48.5, que
representa la pérdida de valor por día para el vendedor). Aquí devolvemos
theta "por año" y también una versión "por día" para comparar directo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Usamos math.erf en vez de scipy para cero dependencias extra.
SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _d1_d2(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float]:
    if T <= 0 or sigma <= 0:
        raise ValueError("T y sigma deben ser positivos")
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def bs_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "CALL") -> float:
    """Precio teórico Black-Scholes (sin dividendos, estilo europeo)."""
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    if option_type.upper() == "CALL":
        return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    elif option_type.upper() == "PUT":
        return K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)
    raise ValueError("option_type debe ser 'CALL' o 'PUT'")


@dataclass
class Greeks:
    delta: float
    gamma: float
    theta_annual: float
    theta_daily: float
    vega: float
    rho: float


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = "CALL") -> Greeks:
    """Griegas cerradas. Vega y Theta se devuelven ya escaladas:
    - vega: cambio en precio por 1.00 (100%) de cambio en sigma (Binance usa la misma convención,
      ojo: algunos proveedores la dan por 1% -> divide entre 100 si necesitas comparar así).
    - theta_daily: cambio en precio por día calendario (T en años / 365).
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    option_type = option_type.upper()

    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))
    vega = S * _norm_pdf(d1) * math.sqrt(T)

    if option_type == "CALL":
        delta = _norm_cdf(d1)
        theta_annual = (
            -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
            - r * K * math.exp(-r * T) * _norm_cdf(d2)
        )
        rho = K * T * math.exp(-r * T) * _norm_cdf(d2)
    elif option_type == "PUT":
        delta = _norm_cdf(d1) - 1
        theta_annual = (
            -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        )
        rho = -K * T * math.exp(-r * T) * _norm_cdf(-d2)
    else:
        raise ValueError("option_type debe ser 'CALL' o 'PUT'")

    return Greeks(
        delta=delta,
        gamma=gamma,
        theta_annual=theta_annual,
        theta_daily=theta_annual / 365,
        vega=vega,
        rho=rho,
    )


def implied_vol(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str = "CALL",
    initial_guess: float = 0.5,
    tol: float = 1e-6,
    max_iter: int = 100,
) -> float | None:
    """
    Invierte Black-Scholes para obtener la IV que reproduce el precio de
    mercado observado. Newton-Raphson (rápido) con fallback a bisección
    (robusto) si Newton no converge o sale de rango.
    """
    # --- Intento 1: Newton-Raphson usando vega como derivada ---
    sigma = initial_guess
    for _ in range(max_iter):
        try:
            price = bs_price(S, K, T, r, sigma, option_type)
            vega = bs_greeks(S, K, T, r, sigma, option_type).vega
        except ValueError:
            break
        diff = price - market_price
        if abs(diff) < tol:
            return sigma
        if vega < 1e-8:
            break  # vega casi cero -> Newton se vuelve inestable, saltamos a bisección
        sigma -= diff / vega
        if sigma <= 0:
            break

    # --- Fallback: bisección en un rango amplio y seguro ---
    lo, hi = 1e-4, 5.0  # 0.01% a 500% de IV anualizada
    price_lo = bs_price(S, K, T, r, lo, option_type) - market_price
    price_hi = bs_price(S, K, T, r, hi, option_type) - market_price
    if price_lo * price_hi > 0:
        return None  # no hay raíz en el rango -> precio de mercado inconsistente con BS

    for _ in range(200):
        mid = (lo + hi) / 2
        price_mid = bs_price(S, K, T, r, mid, option_type) - market_price
        if abs(price_mid) < tol:
            return mid
        if price_lo * price_mid < 0:
            hi = mid
        else:
            lo = mid
            price_lo = price_mid
    return (lo + hi) / 2
