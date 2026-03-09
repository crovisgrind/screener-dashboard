"""
backtest_recuperacao.py
=======================
Sistema de trading inspirado no Qullamaggie (Kristjan Kullamägi),
adaptado para ações do Ibovespa em swing trade de 5–10 dias.

FILTRO DE UNIVERSO:
  - Close > MA50 e Close > MA200         (tendência de alta confirmada)
  - Volume médio 20d > 500k ações        (liquidez mínima)

ENTRADA — VCP (Volatility Contraction Pattern):
  - Contração: range 5d < 50% do range 20d
  - Breakout:  close > máxima dos últimos 10 dias
  - Volume:    volume hoje > 2.0x média 20d
  - Entrada:   abertura do dia seguinte ao sinal
  - Shares:    múltiplos de 100 (lote padrão B3, mercado cheio)

SAÍDA:
  - Alvo:       +7% sobre entrada
  - Stop:       mínima dos últimos 5 dias
  - Break-even: se papel sobe +3.5%, stop sobe para entrada
  - Força:      se no dia 3 ainda abaixo da entrada, sai
  - Time stop:  saída forçada no dia 7

GESTÃO DE RISCO:
  - 1% do capital por trade
  - Máx 3 posições simultâneas
  - Tamanho arredondado para lote de 100
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIGURAÇÃO
# ─────────────────────────────────────────────

ACOES = [
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

PERIODO           = "5y"
CAPITAL_INICIAL   = 100_000
RISK_PER_TRADE    = 0.01
MAX_POSICOES      = 3

MA_FAST           = 50
MA_SLOW           = 200
VOL_MEDIO_MIN     = 500_000
PRECO_MIN         = 5.00        # preço mínimo da ação (evita penny stocks)

CONTRACAO_JANELA  = 5
CONTRACAO_REF     = 20
CONTRACAO_FATOR   = 0.50
BREAKOUT_JANELA   = 10
VOLUME_FATOR      = 2.0

TAKE_PROFIT       = 0.07
BREAKEVEN_GATILHO = 0.035
FORCA_DIAS        = 3
TIME_STOP         = 7


# ─────────────────────────────────────────────
# 1. DOWNLOAD
# ─────────────────────────────────────────────

def baixar_dados():
    tickers = ACOES + ["BOVA11.SA"]
    print(f"\n[1/3] Baixando {len(tickers)} tickers ({PERIODO})...")
    df = yf.download(
        tickers, period=PERIODO, interval="1d",
        progress=True, auto_adjust=True, actions=False, threads=True,
    )
    print(f"    {len(df['Close'])} pregões")
    return df["Close"], df["Open"], df["Volume"], df["High"], df["Low"]


# ─────────────────────────────────────────────
# 2. BACKTEST POR TICKER
# ─────────────────────────────────────────────

def backtest_ticker(ticker, close, open_, volume, high, low, bova_close):
    c = close[ticker].dropna()
    o = open_[ticker].dropna()
    v = volume[ticker].dropna()
    h = high[ticker].dropna()
    l = low[ticker].dropna()

    idx = c.index.intersection(o.index).intersection(v.index) \
               .intersection(h.index).intersection(l.index)
    c = c.loc[idx]
    o = o.loc[idx]
    v = v.loc[idx]
    h = h.loc[idx]
    l = l.loc[idx]

    if len(c) < MA_SLOW + CONTRACAO_REF + TIME_STOP + 5:
        return []

    ma50    = c.rolling(MA_FAST).mean()
    ma200   = c.rolling(MA_SLOW).mean()
    vol_ma  = v.rolling(20).mean()
    r5d     = h.rolling(CONTRACAO_JANELA).max() - l.rolling(CONTRACAO_JANELA).min()
    r20d    = h.rolling(CONTRACAO_REF).max()    - l.rolling(CONTRACAO_REF).min()
    max10d  = h.rolling(BREAKOUT_JANELA).max().shift(1)
    min5d   = l.rolling(CONTRACAO_JANELA).min().shift(1)

    trades  = []
    posicao = None
    inicio  = MA_SLOW + CONTRACAO_REF + 5

    for i in range(inicio, len(c) - 1):
        data    = c.index[i]
        close_h = float(c.iloc[i])
        high_h  = float(h.iloc[i])
        low_h   = float(l.iloc[i])
        vol_h   = float(v.iloc[i])
        ma50_h  = float(ma50.iloc[i])
        ma200_h = float(ma200.iloc[i])
        volma_h = float(vol_ma.iloc[i])
        r5_h    = float(r5d.iloc[i])
        r20_h   = float(r20d.iloc[i])
        max10_h = float(max10d.iloc[i])
        min5_h  = float(min5d.iloc[i])

        open_next     = float(o.iloc[i + 1])
        data_seguinte = c.index[i + 1]

        if any(np.isnan(x) for x in [ma50_h, ma200_h, volma_h, r5_h, r20_h,
                                       max10_h, min5_h, open_next]):
            continue

        # ── GESTÃO DA POSIÇÃO ABERTA ──────────────────────
        if posicao is not None:
            entrada     = posicao["entrada"]
            dias_aberto = (data - posicao["data_entrada"]).days

            # Saídas normais
            saiu   = False
            saida  = None
            motivo = None

            if high_h >= entrada * (1 + TAKE_PROFIT):
                saida  = round(entrada * (1 + TAKE_PROFIT), 2)
                motivo = "take_profit"
                saiu   = True
            elif low_h <= posicao["stop"]:
                saida  = posicao["stop"]
                motivo = "stop_loss"
                saiu   = True
            elif dias_aberto >= TIME_STOP:
                saida  = close_h
                motivo = "time_stop"
                saiu   = True

            if saiu:
                pct = (saida / entrada - 1) * 100
                pnl = (saida - entrada) * posicao["shares"]
                try:
                    rb = (float(bova_close.loc[data]) /
                          float(bova_close.loc[posicao["data_entrada"]]) - 1) * 100
                except:
                    rb = np.nan
                trades.append({
                    "ticker":       ticker.replace(".SA", ""),
                    "data_entrada": posicao["data_entrada"].strftime("%Y-%m-%d"),
                    "data_saida":   data.strftime("%Y-%m-%d"),
                    "dias":         dias_aberto,
                    "entrada":      round(entrada, 2),
                    "saida":        round(saida, 2),
                    "stop_inicial": round(posicao["stop_inicial"], 2),
                    "shares":       int(posicao["shares"]),
                    "pnl":          round(pnl, 2),
                    "pnl_pct":      round(pct, 2),
                    "ret_bova":     round(rb, 2) if not np.isnan(rb) else np.nan,
                    "ret_rel":      round(pct - rb, 2) if not np.isnan(rb) else np.nan,
                    "motivo":       motivo,
                    "vol_ratio":    round(posicao["vol_ratio"], 2),
                    "contracao":    round(posicao["contracao"], 2),
                })
                posicao = None

        # ── BUSCA DE ENTRADA ──────────────────────────────
        if posicao is None:
            if close_h <= ma50_h or close_h <= ma200_h:
                continue
            if volma_h < VOL_MEDIO_MIN:
                continue
            # Preço mínimo
            if close_h < PRECO_MIN:
                continue
            if r20_h <= 0:
                continue
            contracao = r5_h / r20_h
            if contracao >= CONTRACAO_FATOR:
                continue
            if close_h <= max10_h:
                continue
            vol_ratio = vol_h / volma_h if volma_h > 0 else 0
            if vol_ratio < VOLUME_FATOR:
                continue

            stop_price    = min5_h
            preco_entrada = open_next

            risco_share = preco_entrada - stop_price
            if risco_share <= 0:
                continue

            shares = int((CAPITAL_INICIAL * RISK_PER_TRADE / risco_share) // 100) * 100
            if shares < 100:
                continue

            max_shares = int((CAPITAL_INICIAL * 0.33 / preco_entrada) // 100) * 100
            shares     = min(shares, max_shares)
            if shares < 100:
                continue

            posicao = {
                "data_entrada":      data_seguinte,
                "data_sinal":        data,
                "entrada":           preco_entrada,
                "stop":              stop_price,
                "stop_inicial":      stop_price,
                "shares":            shares,
                "vol_ratio":         vol_ratio,
                "contracao":         contracao,
            }

    return trades


# ─────────────────────────────────────────────
# 3. RESUMO
# ─────────────────────────────────────────────

def gerar_resumo(df):
    sep = "=" * 62
    L   = []

    pnl    = df["pnl_pct"]
    rel    = df["ret_rel"].dropna()
    wins   = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    n      = len(pnl)

    winrate    = len(wins) / n
    avg_win    = wins.mean()   if len(wins)   > 0 else 0
    avg_loss   = losses.mean() if len(losses) > 0 else 0
    payoff     = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    pf         = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else np.inf
    expectancy = (winrate * avg_win) + ((1 - winrate) * avg_loss)

    streak = max_streak = 0
    for p in pnl:
        streak     = streak + 1 if p <= 0 else 0
        max_streak = max(max_streak, streak)

    L.append(sep)
    L.append("  BACKTEST — Qullamaggie VCP adaptado para IBOV")
    L.append("  MA50+MA200 | VCP | vol 2x | entrada open D+1 | lote 100")
    L.append(sep)
    L.append(f"  Período         : {df['data_entrada'].min()} → {df['data_saida'].max()}")
    L.append(f"  Total trades    : {n}")
    L.append(f"  Ações únicas    : {df['ticker'].nunique()}")
    L.append(f"  Wins / Losses   : {len(wins)} / {len(losses)}")
    L.append(f"  Winrate         : {winrate*100:.1f}%")
    L.append(f"  Retorno médio   : {pnl.mean():+.2f}%  (mediana {pnl.median():+.2f}%)")
    L.append(f"  Ganho médio     : {avg_win:+.2f}%")
    L.append(f"  Perda média     : {avg_loss:+.2f}%")
    L.append(f"  Payoff ratio    : {payoff:.2f}x")
    L.append(f"  Profit factor   : {pf:.2f}")
    L.append(f"  Expectancy      : {expectancy:+.2f}% por trade")
    L.append(f"  Ret rel. vs BOVA11 : {rel.mean():+.2f}%")
    L.append(f"  Max loss streak : {max_streak} trades seguidos")
    L.append(f"  Duração média   : {df['dias'].mean():.1f} dias")
    L.append(f"  Shares médios   : {df['shares'].mean():.0f} ações (lote 100)")
    L.append(f"  Contração média : {df['contracao'].mean():.2f}x")
    L.append(f"  Volume ratio    : {df['vol_ratio'].mean():.2f}x")
    L.append("")

    L.append(f"  {'─'*58}")
    L.append("  Por motivo de saída:")
    L.append(f"  {'─'*58}")
    L.append(f"  {'Motivo':<15} {'N':>5}  {'Winrate':>8}  {'Ret médio':>10}  {'Duração':>8}")
    for motivo, grp in df.groupby("motivo"):
        w  = (grp["pnl_pct"] > 0).mean() * 100
        rm = grp["pnl_pct"].mean()
        d  = grp["dias"].mean()
        L.append(f"  {motivo:<15} {len(grp):>5}  {w:>7.1f}%  {rm:>+9.2f}%  {d:>7.1f}d")
    L.append("")

    por_acao = (
        df.groupby("ticker")
        .agg(
            n=("pnl_pct", "count"),
            winrate=("pnl_pct", lambda x: (x > 0).mean() * 100),
            ret_medio=("pnl_pct", "mean"),
            ret_rel_medio=("ret_rel", "mean"),
        )
        .round(2)
    )

    L.append(f"  {'─'*58}")
    L.append("  Top 10 ações:")
    L.append(f"  {'─'*58}")
    L.append(por_acao.sort_values("ret_medio", ascending=False).head(10).to_string())

    L.append(f"\n  {'─'*58}")
    L.append("  Bottom 10 ações:")
    L.append(f"  {'─'*58}")
    L.append(por_acao.sort_values("ret_medio").head(10).to_string())

    L.append(f"\n  {'─'*58}")
    L.append("  Distribuição de retornos por trade:")
    L.append(f"  {'─'*58}")
    bins   = [(-np.inf,-10),(-10,-5),(-5,-2),(-2,0),(0,2),(2,5),(5,10),(10,np.inf)]
    labels = ["< -10%","-10 a -5%","-5 a -2%","-2 a 0%","0 a +2%","+2 a +5%","+5 a +10%","> +10%"]
    for (lo, hi), label in zip(bins, labels):
        count = ((pnl > lo) & (pnl <= hi)).sum()
        bar   = "█" * int(count / max(1, n) * 40)
        L.append(f"  {label:>12}  {count:>4}  {bar}")

    L.append(f"\n{sep}")
    return "\n".join(L)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Backtest Qullamaggie VCP — IBOV Swing Trade")
    print("=" * 62)

    close, open_, volume, high, low = baixar_dados()
    bova = close["BOVA11.SA"]

    print(f"\n[2/3] Rodando backtest em {len(ACOES)} ações...")
    todos_trades = []
    for ticker in ACOES:
        if ticker not in close.columns:
            continue
        try:
            trades = backtest_ticker(ticker, close, open_, volume, high, low, bova)
            if trades:
                todos_trades.extend(trades)
                pnls = [t["pnl_pct"] for t in trades]
                wins = sum(1 for p in pnls if p > 0)
                print(f"  {ticker.replace('.SA',''):<10} {len(trades):>3} trades  "
                      f"{wins/len(trades)*100:>5.0f}% acerto  "
                      f"média {np.mean(pnls):>+5.1f}%")
            else:
                print(f"  {ticker.replace('.SA',''):<10}   0 trades")
        except Exception as e:
            print(f"  {ticker.replace('.SA',''):<10} ERRO: {e}")

    print(f"\n[3/3] Consolidando {len(todos_trades)} trades...")

    if not todos_trades:
        print("\n[AVISO] Nenhum trade encontrado. Verifique os parâmetros.")
    else:
        df = pd.DataFrame(todos_trades)
        resumo = gerar_resumo(df)
        print("\n" + resumo)

        df.to_csv("backtest_qullamaggie_trades.csv", index=False, float_format="%.2f")
        with open("backtest_qullamaggie_resumo.txt", "w", encoding="utf-8") as f:
            f.write(resumo)

        print(f"\n  CSV:    backtest_qullamaggie_trades.csv  ({len(df)} trades)")
        print(f"  Resumo: backtest_qullamaggie_resumo.txt")

    print("\nConcluído.")