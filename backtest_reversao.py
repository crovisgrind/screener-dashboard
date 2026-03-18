"""
backtest_reversao_v3.py
=======================
Estratégia: Reversão MRS — versão com whitelist/blacklist

Refinamentos sobre v2:
  - WHITELIST: só entra em ações que historicamente revertem
    (apareceram como top em múltiplos backtests)
  - BLACKLIST: nunca entra em ações que consistentemente falham
  - Mantém: MRS -4% a -5%, ≥3 cruzamentos em 30d, TP +5%
  - Testa: TS5d, TS7d, TS10d

Whitelist derivada dos tops dos backtests v1 e v2:
  BPAC11, CPLE3, CSMG3, CURY3, EGIE3, ITUB4, LWSA3,
  NEOE3, PETR4, PSSA3, RENT3, SBSP3, SLCE3, TOTS3

Blacklist derivada dos bottoms recorrentes:
  B3SA3, BBDC3, CEAB3, CMIG4, CMIN3, COGN3, CVCB3,
  CYRE3, ECOR3, HYPE3, RECV3, VAMO3
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── WHITELIST / BLACKLIST ─────────────────────────────────────────
WHITELIST = [
    "BPAC11.SA","CPLE3.SA","CSMG3.SA","CURY3.SA","EGIE3.SA",
    "ITUB4.SA","LWSA3.SA","NEOE3.SA","PETR4.SA","PSSA3.SA",
    "RENT3.SA","SBSP3.SA","SLCE3.SA","TOTS3.SA",
]

# Para comparação: também testa o universo COMPLETO (sem whitelist)
# e o universo BLACKLIST-REMOVED (remove só as ruins, sem forçar lista)
IBRX100_FULL = [
    "ALOS3.SA","ABEV3.SA","ANIM3.SA","ASAI3.SA","AURE3.SA","AXIA3.SA",
    "AZZA3.SA","B3SA3.SA","BBSE3.SA","BBDC3.SA","BBDC4.SA","BRAP4.SA",
    "BBAS3.SA","BRKM5.SA","BRAV3.SA","BPAC11.SA","CXSE3.SA","BHIA3.SA",
    "CBAV3.SA","CEAB3.SA","CMIG4.SA","COGN3.SA","CSMG3.SA","CPLE3.SA",
    "CSAN3.SA","CPFE3.SA","CMIN3.SA","CURY3.SA","CVCB3.SA","CYRE3.SA",
    "DIRR3.SA","ECOR3.SA","ENGI11.SA","ENEV3.SA","EGIE3.SA","EQTL3.SA",
    "EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GGPS3.SA","GMAT3.SA",
    "HAPV3.SA","HYPE3.SA","IGTI11.SA","IRBR3.SA","ITSA4.SA","ITUB4.SA",
    "KLBN11.SA","RENT3.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","POMO4.SA",
    "BEEF3.SA","MOVI3.SA","MRVE3.SA","MULT3.SA","NATU3.SA","NEOE3.SA",
    "PCAR3.SA","PETR3.SA","PETR4.SA","RECV3.SA","PRIO3.SA","PSSA3.SA",
    "RADL3.SA","RAIZ4.SA","RAPT4.SA","RDOR3.SA","RAIL3.SA","SBSP3.SA",
    "SAPR11.SA","SANB11.SA","SMTO3.SA","CSNA3.SA","SLCE3.SA","SUZB3.SA",
    "TAEE11.SA","VIVT3.SA","TEND3.SA","TIMS3.SA","TOTS3.SA","UGPA3.SA",
    "USIM5.SA","VALE3.SA","VAMO3.SA","VBBR3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA",
]

BLACKLIST_TICKERS = {
    "B3SA3.SA","BBDC3.SA","CEAB3.SA","CMIG4.SA","CMIN3.SA",
    "COGN3.SA","CVCB3.SA","CYRE3.SA","ECOR3.SA","HYPE3.SA",
    "RECV3.SA","VAMO3.SA",
}

IBRX_NO_BLACK = [t for t in IBRX100_FULL if t not in BLACKLIST_TICKERS]

# Três universos para comparar
UNIVERSOS = {
    "WHITELIST":  WHITELIST,
    "NO_BLACK":   IBRX_NO_BLACK,
    "FULL":       IBRX100_FULL,
}

# ── PARÂMETROS ────────────────────────────────────────────────────
PERIODO   = "5y"
LENGTH    = 200
MRS_MIN   = -0.05       # -5%
MRS_MAX   = -0.04       # -4%
OSCIL_WIN = 30
MIN_CRUZ  = 3
TP        = 0.05
TIME_STOPS = [5, 7, 10]

# ── DOWNLOAD ─────────────────────────────────────────────────────
_cache = {}
def baixar(ticker, retries=2):
    if ticker in _cache:
        return _cache[ticker]
    for t in range(retries):
        try:
            df = yf.download(ticker, period=PERIODO, interval="1d",
                             progress=False, auto_adjust=True,
                             actions=False, threads=False)
            if df is None or len(df) == 0:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            _cache[ticker] = df
            return df
        except:
            if t < retries - 1:
                import time; time.sleep(0.4)
    _cache[ticker] = None
    return None

# ── SINAIS ────────────────────────────────────────────────────────
def calcular_sinais(ticker, df, bova_df):
    try:
        c  = df['Close']
        o  = df['Open']
        bc = bova_df['Close']
        idx = c.index.intersection(bc.index).intersection(o.index)
        c = c.loc[idx]; o = o.loc[idx]; bc = bc.loc[idx]
        if len(c) < LENGTH + OSCIL_WIN + 5:
            return []
        rp_c = c / bc
        mrs  = (rp_c / rp_c.rolling(LENGTH).mean() - 1) * 100
        sinais = []
        for i in range(LENGTH + OSCIL_WIN, len(c) - 1):
            mrs_h = float(mrs.iloc[i])
            if np.isnan(mrs_h):
                continue
            if not (MRS_MIN * 100 <= mrs_h <= MRS_MAX * 100):
                continue
            mrs_win = mrs.iloc[i - OSCIL_WIN + 1:i + 1].dropna().values
            if len(mrs_win) < 4:
                continue
            cruzamentos = sum(
                1 for j in range(len(mrs_win) - 1)
                if (mrs_win[j] > 0) != (mrs_win[j+1] > 0)
            )
            if cruzamentos < MIN_CRUZ:
                continue
            open_entrada = float(o.iloc[i + 1])
            if np.isnan(open_entrada) or open_entrada <= 0:
                continue
            sinais.append({
                'data':         c.index[i],
                'data_entrada': c.index[i + 1],
                'open_entrada': open_entrada,
                'mrs_entrada':  round(mrs_h, 3),
                'cruzamentos':  cruzamentos,
            })
        return sinais
    except:
        return []

# ── SIMULA ────────────────────────────────────────────────────────
def simular(ticker, sinais, df, bova_df, time_stop):
    trades = []
    try:
        c   = df['Close']
        bc  = bova_df['Close']
        idx = list(c.index.intersection(bc.index))
        c   = c.loc[idx]; bc = bc.loc[idx]
        c_list = list(c.index)
    except:
        return []

    ultimo = -999
    for sig in sinais:
        de = sig['data_entrada']
        if de not in c.index:
            continue
        pe = c_list.index(de)
        if pe <= ultimo:
            continue
        entrada = sig['open_entrada']
        alvo    = entrada * (1 + TP)
        saida   = None; motivo = None; dias = 0

        for d in range(1, time_stop + 1):
            pd_ = pe + d
            if pd_ >= len(c):
                break
            close_d = float(c.iloc[pd_]); dias = d
            if close_d >= alvo:
                saida = alvo; motivo = 'take_profit'; ultimo = pd_; break
            if d == time_stop:
                saida = close_d; motivo = 'time_stop'; ultimo = pd_; break

        if saida is None:
            continue
        ds  = c.index[min(pe + dias, len(c) - 1)]
        pnl = round((saida / entrada - 1) * 100, 3)
        try:
            rb = round((float(bc.loc[ds]) / float(bc.loc[de]) - 1) * 100, 3)
        except:
            rb = np.nan
        trades.append({
            'ticker':       ticker.replace('.SA',''),
            'data_entrada': de.strftime('%Y-%m-%d'),
            'dias':         dias,
            'mrs_entrada':  sig['mrs_entrada'],
            'cruzamentos':  sig['cruzamentos'],
            'entrada':      round(entrada, 2),
            'pnl_pct':      pnl,
            'ret_bova':     rb,
            'ret_rel':      round(pnl - rb, 3) if not np.isnan(rb) else np.nan,
            'motivo':       motivo,
        })
    return trades

# ── STATS ─────────────────────────────────────────────────────────
def stats(s):
    s = s.dropna()
    if s.empty: return {}
    w = s[s > 0]; l = s[s <= 0]
    pf  = abs(w.sum() / l.sum()) if l.sum() != 0 else np.inf
    exp = (len(w)/len(s))*w.mean() + (len(l)/len(s))*l.mean()
    streak = ms = 0
    for p in s:
        streak = streak+1 if p <= 0 else 0
        ms = max(ms, streak)
    return dict(n=len(s), media=s.mean(), mediana=s.median(),
                acerto=(s>0).mean()*100, pf=pf, exp=exp,
                p25=np.percentile(s,25), p75=np.percentile(s,75),
                ganho=w.mean() if len(w) else 0,
                perda=l.mean() if len(l) else 0, ms=ms)

# ── RELATÓRIO ─────────────────────────────────────────────────────
def relatorio(resultados):
    sep = "=" * 72
    L = [sep,
         "  BACKTEST — REVERSÃO MRS v3 · WHITELIST vs BLACKLIST-FREE vs FULL",
         "  Entrada: MRS -4% a -5% | ≥3 cruzamentos 30d | TP +5%",
         sep]

    # Tabela geral: universo × time stop
    L.append(f"\n  {'UNIVERSO':<14} {'TS':<6} {'N':>5}  {'Win%':>7}  {'RetMed':>8}  {'PF':>6}  {'Exp':>8}  {'Dur':>5}  {'MaxStr':>7}")
    L.append(f"  {'─'*72}")

    for univ_nome, ts_dict in resultados.items():
        for ts, data in sorted(ts_dict.items()):
            df_u = data['df']; s = data['s']
            if not s: continue
            L.append(
                f"  {univ_nome:<14} TS{ts}d  {s['n']:>5}  {s['acerto']:>6.1f}%"
                f"  {s['media']:>+7.2f}%  {s['pf']:>6.2f}"
                f"  {s['exp']:>+7.2f}%  {df_u['dias'].mean():>4.1f}d"
                f"  {s['ms']:>7}"
            )
        L.append(f"  {'·'*72}")

    # Destaque WHITELIST detalhado
    for ts, data in sorted(resultados.get('WHITELIST',{}).items()):
        df_u = data['df']; s = data['s']
        if not s: continue
        pnl = df_u['pnl_pct']
        L.append(f"\n  {'─'*70}")
        L.append(f"  WHITELIST — TS{ts}d")
        L.append(f"  {'─'*70}")
        L.append(f"  Trades / Ações   : {s['n']} / {df_u['ticker'].nunique()}")
        L.append(f"  Winrate          : {s['acerto']:.1f}%")
        L.append(f"  Ret médio/mediana: {s['media']:+.2f}% / {s['mediana']:+.2f}%")
        L.append(f"  Ganho / Perda    : {s['ganho']:+.2f}% / {s['perda']:+.2f}%")
        L.append(f"  Profit factor    : {s['pf']:.2f}")
        L.append(f"  Expectancy       : {s['exp']:+.2f}%")
        L.append(f"  Ret rel BOVA11   : {df_u['ret_rel'].dropna().mean():+.2f}%")
        L.append(f"  P25 / P75        : {s['p25']:+.1f}% / {s['p75']:+.1f}%")
        L.append(f"  Max loss streak  : {s['ms']}")

        # Por motivo
        L.append(f"\n    Por motivo:")
        L.append(f"    {'Motivo':<14} {'N':>5}  {'Win%':>7}  {'RetMed':>10}  {'Dur':>6}")
        for mot, grp in df_u.groupby('motivo'):
            L.append(f"    {mot:<14} {len(grp):>5}  {(grp['pnl_pct']>0).mean()*100:>6.1f}%"
                     f"  {grp['pnl_pct'].mean():>+9.2f}%  {grp['dias'].mean():>5.1f}d")

        # Por ação
        por_acao = (
            df_u.groupby('ticker')
            .agg(n=('pnl_pct','count'),
                 winrate=('pnl_pct', lambda x:(x>0).mean()*100),
                 ret_medio=('pnl_pct','mean'),
                 ret_rel=('ret_rel','mean'))
            .round(2)
        )
        L.append(f"\n    Por ação (whitelist completa):")
        L.append(por_acao.sort_values('ret_medio', ascending=False).to_string())

        # Distribuição
        bins = [(-np.inf,-5),(-5,-2),(-2,0),(0,2),(2,5),(5,np.inf)]
        lbls = ["< −5%","−5 a −2%","−2 a 0%","0 a +2%","+2 a +5%","> +5%"]
        L.append(f"\n    Distribuição:")
        for (lo,hi), lbl in zip(bins, lbls):
            cnt = ((pnl>lo)&(pnl<=hi)).sum()
            L.append(f"    {lbl:>10}  {cnt:>4}  {'█'*int(cnt/max(1,len(pnl))*36)}")

    L.append(f"\n{sep}")
    return "\n".join(L)

# ── MAIN ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 72)
    print("  Backtest Reversão MRS v3 — Whitelist/Blacklist")
    print("=" * 72)

    print("\n[1/3] Baixando BOVA11...")
    bova = baixar("BOVA11.SA")
    if bova is None:
        print("ERRO: BOVA11 não encontrado"); exit(1)

    # Todos os tickers únicos nos 3 universos
    todos_tickers = set()
    for tks in UNIVERSOS.values():
        todos_tickers.update(tks)

    print(f"\n[2/3] Baixando e simulando {len(todos_tickers)} tickers únicos...")

    # Pré-calcula sinais por ticker (reutiliza entre universos)
    sinais_cache = {}
    for i, ticker in enumerate(sorted(todos_tickers), 1):
        print(f"  [{i:>2}/{len(todos_tickers)}] {ticker.replace('.SA',''):<10}", end=' ')
        df = baixar(ticker)
        if df is None:
            print("sem dados"); sinais_cache[ticker] = []; continue
        s = calcular_sinais(ticker, df, bova)
        sinais_cache[ticker] = s
        print(f"{len(s)} sinais")

    print(f"\n[3/3] Simulando 3 universos × 3 time stops...")
    resultados = {}

    for univ_nome, tickers in UNIVERSOS.items():
        resultados[univ_nome] = {}
        for ts in TIME_STOPS:
            todos = []
            for ticker in tickers:
                s = sinais_cache.get(ticker, [])
                if not s: continue
                df = _cache.get(ticker)
                if df is None: continue
                trades = simular(ticker, s, df, bova, ts)
                todos.extend(trades)
            df_ts = pd.DataFrame(todos) if todos else pd.DataFrame()
            st = stats(df_ts['pnl_pct']) if not df_ts.empty else {}
            resultados[univ_nome][ts] = {'df': df_ts, 's': st}
            n = len(df_ts)
            pf = f"{st['pf']:.2f}" if st else "—"
            exp = f"{st['exp']:+.2f}%" if st else "—"
            print(f"  {univ_nome:<14} TS{ts}d → {n:>4} trades | PF {pf} | Exp {exp}")

    relat = relatorio(resultados)
    print("\n" + relat)

    # Salva CSV da whitelist
    for ts, data in resultados.get('WHITELIST',{}).items():
        df_u = data['df']
        if not df_u.empty:
            df_u.to_csv(f"reversao_whitelist_ts{ts}d.csv",
                        index=False, float_format="%.3f")

    with open("backtest_reversao_v3_resumo.txt","w",encoding="utf-8") as f:
        f.write(relat)
    print("\n  CSV: reversao_whitelist_ts*.csv")
    print("  Resumo: backtest_reversao_v3_resumo.txt")
    print("\nConcluído.")