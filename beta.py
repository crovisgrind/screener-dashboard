"""
analise_beta.py
===============
Calcula o beta de cada ação da whitelist vs IBOV (^BVSP)
e compara com o universo completo para identificar padrões.

Uso:
  pip install yfinance pandas numpy scipy
  python analise_beta.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────

WHITELIST = [
    "COGN3.SA", "CMIN3.SA", "BRAP4.SA", "TAEE11.SA",
    "CSNA3.SA", "UGPA3.SA", "CPFE3.SA", "HYPE3.SA",
]

UNIVERSO = [
    "ALOS3.SA","ABEV3.SA","ASAI3.SA","AURE3.SA","AZZA3.SA","B3SA3.SA","BBSE3.SA",
    "BBDC3.SA","BBDC4.SA","BRAP4.SA","BBAS3.SA","BRKM5.SA","BRAV3.SA","BPAC11.SA",
    "BHIA3.SA","CMIG4.SA","COGN3.SA","CPLE3.SA","CSAN3.SA","CPFE3.SA","CMIN3.SA",
    "CURY3.SA","CVCB3.SA","CYRE3.SA","ECOR3.SA","ENGI11.SA","ENEV3.SA","EGIE3.SA",
    "EQTL3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GGPS3.SA","HAPV3.SA","HYPE3.SA",
    "IGTI11.SA","IRBR3.SA","ITSA4.SA","ITUB4.SA","KLBN11.SA","RENT3.SA","LREN3.SA",
    "MGLU3.SA","BEEF3.SA","MOVI3.SA","MRVE3.SA","MULT3.SA","NATU3.SA","NEOE3.SA",
    "PETR3.SA","PETR4.SA","RECV3.SA","PRIO3.SA","PSSA3.SA","RADL3.SA","RAIZ4.SA",
    "RDOR3.SA","RAIL3.SA","SBSP3.SA","SANB11.SA","CSNA3.SA","SLCE3.SA","SUZB3.SA",
    "TAEE11.SA","VIVT3.SA","TIMS3.SA","TOTS3.SA","UGPA3.SA","USIM5.SA","VALE3.SA",
    "VBBR3.SA","WEGE3.SA","YDUQ3.SA",
]

PERIODO = "5y"
BENCHMARK = "^BVSP"


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────

def baixar_dados():
    todos = list(set(UNIVERSO + [BENCHMARK]))
    print(f"Baixando {len(todos)} tickers ({PERIODO})...")
    df = yf.download(todos, period=PERIODO, interval="1d",
                     progress=True, auto_adjust=True, actions=False)
    return df["Close"]


# ─────────────────────────────────────────────
# CÁLCULO DE BETA
# ─────────────────────────────────────────────

def calcular_beta(retornos_acao, retornos_bench):
    """Beta via regressão linear simples."""
    df = pd.concat([retornos_acao, retornos_bench], axis=1).dropna()
    if len(df) < 60:
        return np.nan, np.nan, np.nan
    slope, intercept, r, p, _ = stats.linregress(df.iloc[:, 1], df.iloc[:, 0])
    return round(slope, 3), round(r**2, 3), round(p, 4)


def calcular_todos_betas(close):
    bench_ret = close[BENCHMARK].pct_change().dropna()
    resultados = []

    for ticker in UNIVERSO:
        if ticker not in close.columns:
            continue
        ret = close[ticker].pct_change().dropna()
        beta, r2, p = calcular_beta(ret, bench_ret)
        resultados.append({
            "ticker":    ticker.replace(".SA", ""),
            "beta":      beta,
            "r2":        r2,
            "p_valor":   p,
            "whitelist": ticker in WHITELIST,
        })

    return pd.DataFrame(resultados).sort_values("beta")


# ─────────────────────────────────────────────
# ANÁLISE
# ─────────────────────────────────────────────

def analisar(df):
    sep = "=" * 56
    L   = []

    wl  = df[df["whitelist"] == True]
    out = df[df["whitelist"] == False]

    L.append(sep)
    L.append("  ANÁLISE DE BETA — Whitelist vs Universo")
    L.append(sep)

    L.append(f"\n  {'─'*54}")
    L.append("  WHITELIST — Beta individual")
    L.append(f"  {'─'*54}")
    L.append(f"  {'Ticker':<10} {'Beta':>6}  {'R²':>6}  {'Classificação'}")
    L.append(f"  {'─'*54}")
    for _, row in wl.sort_values("beta").iterrows():
        classe = classificar_beta(row["beta"])
        L.append(f"  {row['ticker']:<10} {row['beta']:>6.3f}  {row['r2']:>6.3f}  {classe}")

    L.append(f"\n  Beta médio whitelist  : {wl['beta'].mean():.3f}")
    L.append(f"  Beta mediana whitelist: {wl['beta'].median():.3f}")
    L.append(f"  Beta médio universo   : {out['beta'].mean():.3f}")
    L.append(f"  Beta mediana universo : {out['beta'].median():.3f}")

    # Distribuição por faixa
    L.append(f"\n  {'─'*54}")
    L.append("  Distribuição por faixa de beta")
    L.append(f"  {'─'*54}")
    faixas = [
        ("Defensivo   (< 0.7)",  lambda b: b < 0.7),
        ("Moderado    (0.7–1.0)", lambda b: 0.7 <= b < 1.0),
        ("Neutro      (1.0–1.3)", lambda b: 1.0 <= b < 1.3),
        ("Agressivo   (≥ 1.3)",  lambda b: b >= 1.3),
    ]
    L.append(f"  {'Faixa':<25} {'Whitelist':>10}  {'Universo':>10}")
    for label, cond in faixas:
        n_wl  = wl[wl["beta"].apply(cond)].shape[0]
        n_out = out[out["beta"].apply(cond)].shape[0]
        pct_wl  = n_wl  / len(wl)  * 100 if len(wl)  > 0 else 0
        pct_out = n_out / len(out) * 100 if len(out) > 0 else 0
        L.append(f"  {label:<25} {n_wl:>3} ({pct_wl:>4.0f}%)  {n_out:>3} ({pct_out:>4.0f}%)")

    # Ranking completo do universo
    L.append(f"\n  {'─'*54}")
    L.append("  Ranking completo (ordenado por beta)")
    L.append(f"  {'─'*54}")
    L.append(f"  {'Ticker':<10} {'Beta':>6}  {'R²':>6}  {'WL':>4}  {'Classificação'}")
    L.append(f"  {'─'*54}")
    for _, row in df.sort_values("beta").iterrows():
        wl_mark = "✓" if row["whitelist"] else " "
        classe  = classificar_beta(row["beta"])
        L.append(f"  {row['ticker']:<10} {row['beta']:>6.3f}  {row['r2']:>6.3f}  {wl_mark:>4}  {classe}")

    L.append(f"\n{sep}")
    L.append("  Legenda R²: correlação com IBOV (1.0 = movimento idêntico)")
    L.append("  Beta < 1: menos volátil que o mercado")
    L.append("  Beta = 1: move igual ao mercado")
    L.append("  Beta > 1: mais volátil que o mercado")
    L.append(sep)

    return "\n".join(L)


def classificar_beta(b):
    if pd.isna(b):      return "sem dados"
    if b < 0:           return "inverso"
    if b < 0.5:         return "muito defensivo"
    if b < 0.7:         return "defensivo"
    if b < 1.0:         return "moderado"
    if b < 1.3:         return "neutro/mercado"
    if b < 1.6:         return "agressivo"
    return "muito agressivo"


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    close   = baixar_dados()
    df_beta = calcular_todos_betas(close)
    resumo  = analisar(df_beta)

    print("\n" + resumo)

    df_beta.to_csv("analise_beta.csv", index=False, float_format="%.3f")
    with open("analise_beta.txt", "w", encoding="utf-8") as f:
        f.write(resumo)

    print("\n  CSV: analise_beta.csv")
    print("  Texto: analise_beta.txt")
    print("\nConcluído.")