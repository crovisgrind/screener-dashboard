"""
backtest_apgar_v4.py
====================
Grid de saídas — APGAR Score · IBRX100

BASE: score 5–8 + B=2 + H2 (MRS>0 + RSV>0) — melhor variação da v3

Dimensão 1 — Time stop: 5, 7, 10, 15 dias
Dimensão 2 — Score dinâmico: sem / sai se score cai abaixo de 4 / abaixo de 3

Total: 4 × 3 = 12 combinações

Lógica do score dinâmico:
  A cada dia em posição, recalcula o score.
  Se o score cair abaixo do limiar → sai no fechamento desse dia.
  Interpretação: "o setup se deteriorou, o mercado mudou de opinião"

Uso:
  pip install yfinance pandas numpy
  python backtest_apgar_v4.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

ACOES = [
    "ALOS3.SA","ABEV3.SA","ANIM3.SA","ASAI3.SA","AURE3.SA","AXIA3.SA","AXIA6.SA","AXIA7.SA",
    "AZZA3.SA","B3SA3.SA","BBSE3.SA","BBDC3.SA","BBDC4.SA","BRAP4.SA","BBAS3.SA","BRKM5.SA",
    "BRAV3.SA","BPAC11.SA","CXSE3.SA","BHIA3.SA","CBAV3.SA","CEAB3.SA","CMIG4.SA","COGN3.SA",
    "CSMG3.SA","CPLE3.SA","CSAN3.SA","CPFE3.SA","CMIN3.SA","CURY3.SA","CVCB3.SA","CYRE3.SA",
    "CYRE4.SA","DIRR3.SA","ECOR3.SA","EMBJ3.SA","ENGI11.SA","ENEV3.SA","EGIE3.SA","EQTL3.SA",
    "EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GGPS3.SA","GMAT3.SA","HAPV3.SA","HYPE3.SA",
    "IGTI11.SA","INTB3.SA","IRBR3.SA","ISAE4.SA","ITSA4.SA","ITUB4.SA","KLBN11.SA","RENT3.SA",
    "RENT4.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","POMO4.SA","MBRF3.SA","BEEF3.SA","MOTV3.SA",
    "MOVI3.SA","MRVE3.SA","MULT3.SA","NATU3.SA","NEOE3.SA","PCAR3.SA","PETR3.SA","PETR4.SA",
    "RECV3.SA","PRIO3.SA","AUAU3.SA","PSSA3.SA","RADL3.SA","RAIZ4.SA","RAPT4.SA","RDOR3.SA",
    "RAIL3.SA","SBSP3.SA","SAPR11.SA","SANB11.SA","SMTO3.SA","CSNA3.SA","SIMH3.SA","SLCE3.SA",
    "SMFT3.SA","SUZB3.SA","TAEE11.SA","VIVT3.SA","TEND3.SA","TIMS3.SA","TOTS3.SA","UGPA3.SA",
    "USIM5.SA","VALE3.SA","VAMO3.SA","VBBR3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA",
]

PERIODO = "5y"
LENGTH           = 200
MA_FAST          = 50
MA_MED           = 150
MA_SLOW          = 200
CONTRACAO_JANELA = 5
CONTRACAO_REF    = 20
BREAKOUT_JANELA  = 10
SCORE_MIN        = 5
SCORE_MAX        = 8
TAKE_PROFIT      = 0.05
TIME_STOPS       = [5, 7, 10, 15]
SCORE_LIMIARES   = [None, 4, 3]


def baixar_dados():
    tickers = ACOES + ["BOVA11.SA"]
    print(f"\n[1/3] Baixando {len(tickers)} tickers ({PERIODO})...")
    df = yf.download(tickers, period=PERIODO, interval="1d",
                     progress=True, auto_adjust=True, actions=False, threads=True)
    print(f"    {len(df['Close'])} pregões")
    return df["Close"], df["Open"], df["Volume"], df["High"], df["Low"]


def calcular_indicadores(ticker, close, open_, volume, high, low, bc, bv):
    try:
        c = close[ticker].dropna(); o = open_[ticker].dropna()
        v = volume[ticker].dropna(); h = high[ticker].dropna(); l = low[ticker].dropna()
        bc_s = bc.dropna(); bv_s = bv.dropna()

        idx = c.index.intersection(o.index).intersection(v.index)\
                     .intersection(h.index).intersection(l.index)\
                     .intersection(bc_s.index).intersection(bv_s.index)
        c=c.loc[idx]; o=o.loc[idx]; v=v.loc[idx]; h=h.loc[idx]; l=l.loc[idx]
        bc_s=bc_s.loc[idx]; bv_s=bv_s.loc[idx]

        minimo = MA_SLOW + CONTRACAO_REF + 5
        if len(c) < minimo:
            return None

        ma50    = c.rolling(MA_FAST).mean()
        ma150   = c.rolling(MA_MED).mean()
        ma200   = c.rolling(MA_SLOW).mean()
        vol_ma  = v.rolling(20).mean()
        r5d     = h.rolling(CONTRACAO_JANELA).max() - l.rolling(CONTRACAO_JANELA).min()
        r20d    = h.rolling(CONTRACAO_REF).max()    - l.rolling(CONTRACAO_REF).min()
        max10d  = h.rolling(BREAKOUT_JANELA).max().shift(1)
        ma200_20= ma200.shift(20)
        rp_c    = c / bc_s; rp_v = v / bv_s
        mrs     = (rp_c / rp_c.rolling(LENGTH).mean() - 1) * 100
        rsv     = (rp_v / rp_v.rolling(LENGTH).mean() - 1) * 100

        records = []
        for i in range(minimo, len(c) - 1):
            p=float(c.iloc[i]); m50=float(ma50.iloc[i]); m150=float(ma150.iloc[i])
            m200=float(ma200.iloc[i]); vh=float(v.iloc[i]); vma=float(vol_ma.iloc[i])
            r5=float(r5d.iloc[i]); r20=float(r20d.iloc[i]); mx10=float(max10d.iloc[i])
            mrs_h=float(mrs.iloc[i]); rsv_h=float(rsv.iloc[i])
            m200p=float(ma200_20.iloc[i]); onx=float(o.iloc[i+1])

            if any(np.isnan(x) for x in [m50,m150,m200,vma,r5,r20,mx10,mrs_h,rsv_h,onx]):
                continue

            vr  = vh/vma if vma>0 else 0
            con = r5/r20 if r20>0 else 1
            dm2 = (p/m200-1)*100 if m200>0 else 0
            brk = p > mx10
            ms  = (not np.isnan(m200p)) and (m200>m200p)

            s = {}
            if p>m50>m150>m200 and ms: s['T']=2
            elif p>m50 and p>m200:     s['T']=1
            else:                       s['T']=0
            if p<m200:    s['E']=0
            elif dm2<=25: s['E']=2
            else:         s['E']=1
            if mrs_h>5:    s['R']=2
            elif mrs_h>=0: s['R']=1
            else:          s['R']=0
            if con<0.35:   s['C']=2
            elif con<0.50: s['C']=1
            else:          s['C']=0
            if brk and vr>=2.0:  s['B']=2
            elif brk or vr>=1.5: s['B']=1
            else:                s['B']=0

            records.append({
                'data':c.index[i], 'data_entrada':c.index[i+1],
                'open_entrada':onx, 'score':sum(s.values()),
                'B':s['B'], 'mrs':round(mrs_h,2), 'rsv':round(rsv_h,2),
            })

        return pd.DataFrame(records) if records else None
    except Exception as e:
        print(f"  [ERR] {ticker}: {e}"); return None


def simular(ticker, df_ind, close, bova_close, time_stop, score_limiar):
    trades=[]; c=close[ticker].dropna(); bc=bova_close.dropna(); c_idx=list(c.index)
    sc_map=dict(zip(df_ind['data_entrada'], df_ind['score']))
    ultimo=-999

    for _, row in df_ind.iterrows():
        sc=row['score']
        if sc<SCORE_MIN or sc>SCORE_MAX: continue
        if row['B']!=2: continue
        if row['mrs']<=0 or row['rsv']<=0: continue

        de=row['data_entrada']
        if de not in c.index: continue
        pe=c_idx.index(de)
        if pe<=ultimo: continue

        entrada=row['open_entrada']
        if np.isnan(entrada) or entrada<=0: continue

        alvo=entrada*(1+TAKE_PROFIT)
        saida=None; motivo=None; dias=0; sc_s=sc

        for d in range(1, time_stop+1):
            pd_=pe+d
            if pd_>=len(c): break
            dd=c.index[pd_]; cd=float(c.iloc[pd_]); dias=d
            sc_d=sc_map.get(dd, None)

            if cd>=alvo:
                saida=alvo; motivo='take_profit'; sc_s=sc_d if sc_d else sc
                ultimo=pd_; break

            if score_limiar is not None and sc_d is not None and sc_d<score_limiar:
                saida=cd; motivo=f'score_exit_{score_limiar}'; sc_s=sc_d
                ultimo=pd_; break

            if d==time_stop:
                saida=cd; motivo='time_stop'; sc_s=sc_d if sc_d else sc
                ultimo=pd_; break

        if saida is None: continue

        is_=min(pe+dias, len(c)-1); ds=c.index[is_]
        pnl=round((saida/entrada-1)*100, 2)
        try: rb=round((float(bc.loc[ds])/float(bc.loc[de])-1)*100, 2)
        except: rb=np.nan

        sl_str=str(score_limiar) if score_limiar else 'X'
        trades.append({
            'combo':        f"T{time_stop}_S{sl_str}",
            'time_stop':    time_stop,
            'score_limiar': sl_str,
            'ticker':       ticker.replace('.SA',''),
            'data_entrada': de.strftime('%Y-%m-%d'),
            'data_saida':   ds.strftime('%Y-%m-%d'),
            'dias':         dias,
            'entrada':      round(entrada,2),
            'saida':        round(saida,2),
            'pnl_pct':      pnl,
            'ret_bova':     rb,
            'ret_rel':      round(pnl-rb,2) if not np.isnan(rb) else np.nan,
            'motivo':       motivo,
            'score_entrada':int(sc),
            'score_saida':  int(sc_s) if sc_s is not None else sc,
        })
    return trades


def stats(s):
    s=s.dropna()
    if s.empty: return {}
    w=s[s>0]; l=s[s<=0]
    pf=abs(w.sum()/l.sum()) if l.sum()!=0 else np.inf
    pay=abs(w.mean()/l.mean()) if len(l)>0 and l.mean()!=0 else np.inf
    return {
        'n':len(s),'media':s.mean(),'mediana':s.median(),
        'acerto':(s>0).mean()*100,'payoff':pay,'pf':pf,
        'exp':(len(w)/len(s))*w.mean()+(len(l)/len(s))*l.mean(),
        'p25':np.percentile(s,25),'p75':np.percentile(s,75),
    }

def gerar_resumo(df_todos):
    sep="="*72
    L=[sep,
       "  BACKTEST — APGAR Grid de Saídas · IBRX100",
       "  Entrada: score 5–8 + B=2 + MRS>0 + RSV>0 | TP +5%",
       "  Grid: time stop {5,7,10,15d} × score limiar {sem,<4,<3}",
       sep]

    L.append(f"\n  {'COMBO':<13} {'N':>5}  {'Win%':>7}  {'RetMed':>8}  {'Mediana':>8}  {'PF':>6}  {'Exp':>8}  {'Dur':>6}  {'Str':>5}")
    L.append(f"  {'─'*74}")

    resultados=[]
    for ts in TIME_STOPS:
        for sl in SCORE_LIMIARES:
            lbl=f"T{ts}d S<{sl}" if sl else f"T{ts}d    "
            combo=f"T{ts}_S{sl if sl else 'X'}"
            sub=df_todos[df_todos['combo']==combo] if not df_todos.empty else pd.DataFrame()
            if sub.empty: L.append(f"  {lbl:<13} —"); continue

            s=stats(sub['pnl_pct'])
            stk=mx=0
            for p in sub['pnl_pct']:
                stk=stk+1 if p<=0 else 0; mx=max(mx,stk)

            resultados.append({'combo':combo,'label':lbl,'s':s,
                                'dur':sub['dias'].mean(),'streak':mx,'df':sub})
            L.append(
                f"  {lbl:<13} {s['n']:>5}  {s['acerto']:>6.1f}%"
                f"  {s['media']:>+7.2f}%  {s['mediana']:>+7.2f}%"
                f"  {s['pf']:>6.2f}  {s['exp']:>+7.2f}%"
                f"  {sub['dias'].mean():>5.1f}d  {mx:>5}"
            )
        L.append(f"  {'·'*74}")

    if not resultados:
        L.append("\n  Sem resultados."); return "\n".join(L)

    # Top por PF e Expectancy
    by_pf  = sorted(resultados, key=lambda x: x['s']['pf'],  reverse=True)
    by_exp = sorted(resultados, key=lambda x: x['s']['exp'], reverse=True)

    L.append(f"\n  TOP 3 por Profit Factor:")
    for i,r in enumerate(by_pf[:3],1):
        L.append(f"  {i}. {r['label']} → PF {r['s']['pf']:.2f} | Exp {r['s']['exp']:+.2f}% | N={r['s']['n']}")

    L.append(f"\n  TOP 3 por Expectancy:")
    for i,r in enumerate(by_exp[:3],1):
        L.append(f"  {i}. {r['label']} → Exp {r['s']['exp']:+.2f}% | PF {r['s']['pf']:.2f} | N={r['s']['n']}")

    # Detalhe dos top 4 por PF
    for r in by_pf[:4]:
        sub=r['df']; pnl=sub['pnl_pct']; s=r['s']
        rel=sub['ret_rel'].dropna()
        L.append(f"\n  {'─'*70}")
        L.append(f"  DETALHE: {r['label']}")
        L.append(f"  {'─'*70}")
        L.append(f"  Trades / Ações   : {s['n']} / {sub['ticker'].nunique()}")
        L.append(f"  Winrate          : {s['acerto']:.1f}%")
        L.append(f"  Ret médio/mediana: {s['media']:+.2f}% / {s['mediana']:+.2f}%")
        L.append(f"  Ganho / Perda    : {pnl[pnl>0].mean():+.2f}% / {pnl[pnl<=0].mean():+.2f}%")
        L.append(f"  Payoff / PF      : {s['payoff']:.2f}x / {s['pf']:.2f}")
        L.append(f"  Expectancy       : {s['exp']:+.2f}%")
        L.append(f"  Ret rel BOVA11   : {rel.mean():+.2f}%")
        L.append(f"  P25 / P75        : {s['p25']:+.1f}% / {s['p75']:+.1f}%")

        L.append(f"\n    Motivo:")
        L.append(f"    {'Motivo':<22} {'N':>5}  {'Win%':>7}  {'RetMed':>10}  {'Dur':>6}")
        for mot, grp in sub.groupby('motivo'):
            L.append(f"    {mot:<22} {len(grp):>5}  {(grp['pnl_pct']>0).mean()*100:>6.1f}%"
                     f"  {grp['pnl_pct'].mean():>+9.2f}%  {grp['dias'].mean():>5.1f}d")

        L.append(f"\n    Score de entrada:")
        for sc, grp in sub.groupby('score_entrada'):
            L.append(f"    {sc}  N={len(grp):>3}  Win={( grp['pnl_pct']>0).mean()*100:>5.1f}%"
                     f"  RetMed={grp['pnl_pct'].mean():>+6.2f}%")

        por_acao=(
            sub.groupby('ticker')
            .agg(n=('pnl_pct','count'),
                 winrate=('pnl_pct',lambda x:(x>0).mean()*100),
                 ret_medio=('pnl_pct','mean'),
                 ret_rel=('ret_rel','mean'))
            .query("n>=3").round(2)
        )
        if not por_acao.empty:
            L.append(f"\n    Top ações (≥3):")
            L.append(por_acao.sort_values('ret_medio',ascending=False).head(6).to_string())
            L.append(f"\n    Bottom ações (≥3):")
            L.append(por_acao.sort_values('ret_medio').head(5).to_string())

        bins=[(-np.inf,-5),(-5,-2),(-2,0),(0,2),(2,5),(5,np.inf)]
        lbls=["< −5%","−5 a −2%","−2 a 0%","0 a +2%","+2 a +5%","> +5%"]
        L.append(f"\n    Distribuição:")
        for (lo,hi),lab in zip(bins,lbls):
            cnt=((pnl>lo)&(pnl<=hi)).sum()
            L.append(f"    {lab:>10}  {cnt:>4}  {'█'*int(cnt/max(1,len(pnl))*36)}")

    L.append(f"\n{sep}")
    return "\n".join(L)


if __name__=="__main__":
    print("="*72)
    print("  Backtest APGAR v4 — Grid de Saídas · IBRX100")
    print("="*72)

    close, open_, volume, high, low = baixar_dados()
    bc=close["BOVA11.SA"]; bv=volume["BOVA11.SA"]

    print(f"\n[2/3] Calculando {len(TIME_STOPS)*len(SCORE_LIMIARES)} combinações...")
    todos=[]

    for ticker in ACOES:
        if ticker not in close.columns: continue
        df_ind=calcular_indicadores(ticker, close, open_, volume, high, low, bc, bv)
        if df_ind is None or df_ind.empty:
            print(f"  {ticker.replace('.SA',''):<10} sem dados"); continue

        for ts in TIME_STOPS:
            for sl in SCORE_LIMIARES:
                todos.extend(simular(ticker, df_ind, close, bc, ts, sl))

        base=[t for t in todos if t['ticker']==ticker.replace('.SA','') and t['combo']=='T10_SX']
        print(f"  {ticker.replace('.SA',''):<10} base={len(base):>3}")

    print(f"\n[3/3] Consolidando {len(todos)} trades totais...")
    df_todos=pd.DataFrame(todos) if todos else pd.DataFrame()

    if not df_todos.empty:
        for combo in sorted(df_todos['combo'].unique()):
            n=len(df_todos[df_todos['combo']==combo])
            print(f"  {combo}: {n}")

    resumo=gerar_resumo(df_todos)
    print("\n"+resumo)

    if not df_todos.empty:
        df_todos.to_csv("backtest_apgar_v4_todos.csv", index=False, float_format="%.2f")
        pf_por_combo=df_todos.groupby('combo').apply(
            lambda g: abs(g[g['pnl_pct']>0]['pnl_pct'].sum()/g[g['pnl_pct']<=0]['pnl_pct'].sum())
            if (g['pnl_pct']<=0).any() else 0
        )
        melhor=pf_por_combo.idxmax()
        df_todos[df_todos['combo']==melhor].to_csv(
            f"backtest_apgar_v4_melhor_{melhor}.csv", index=False, float_format="%.2f")
        print(f"\n  Melhor: {melhor}")

    with open("backtest_apgar_v4_resumo.txt","w",encoding="utf-8") as f:
        f.write(resumo)
    print("  CSV: backtest_apgar_v4_todos.csv")
    print("  Resumo: backtest_apgar_v4_resumo.txt")
    print("\nConcluído.")