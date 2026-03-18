# api/screener.py
# Screener unificado: MRS/RSV (sinais de cruzamento) + APGAR (score 0-10)
# Uma única chamada calcula tudo para cada ação.

from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import os

ACOES_PRINCIPAIS = [
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

# ── Parâmetros ────────────────────────────────────────────

# ── Whitelist Reversão MRS ────────────────────────────────────────
# Ações que historicamente revertem quando MRS entre -4% e -5%
# (derivada de backtests v1, v2, v3 — PF 6.32 no TS10d)
REVERSAO_WHITELIST = {
    "BPAC11","CPLE3","CSMG3","CURY3","EGIE3",
    "ITUB4","LWSA3","NEOE3","PETR4","PSSA3",
    "RENT3","SBSP3","SLCE3","TOTS3",
}
REVERSAO_OSCIL_WIN = 30
REVERSAO_MIN_CRUZ  = 3
REVERSAO_MRS_MIN   = -5.0   # %
REVERSAO_MRS_MAX   = -4.0   # %

LENGTH           = 200
MA_FAST          = 50
MA_MED           = 150
MA_SLOW          = 200
CONTRACAO_JANELA = 5
CONTRACAO_REF    = 20
BREAKOUT_JANELA  = 10

# ── Cache ─────────────────────────────────────────────────
_cache = {'data': None, 'data_proc': None, 'processando': False}

def data_pregao():
    # ── Sinais VCP (APGAR) ──────────────────────────────────
    vcp_sinais = [
        {'ticker': a['ticker'], 'tipo': 'VCP', 'preco': a['preco']}
        for a in apgar_lista
        if 5 <= a['score'] <= 8
        and a['scores'].get('breakout') == 2
        and a['mrs'] > 0
        and a['rsv'] > 0
    ]

    agora_br = datetime.utcnow() - timedelta(hours=3)
    if agora_br.time() < time(18, 30):
        return (agora_br - timedelta(days=1)).date()
    return agora_br.date()

def cache_valido():
    return _cache['data'] is not None and _cache['data_proc'] == data_pregao()

# ── Download ──────────────────────────────────────────────
def baixar(ticker, retries=2):
    for t in range(retries):
        try:
            df = yf.download(ticker, period='2y', interval='1d',
                             progress=False, auto_adjust=True,
                             prepost=False, actions=False, threads=False)
            if df is None or len(df) == 0:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) < MA_SLOW + 30:
                return None
            return df
        except Exception as e:
            print(f"[ERR] {ticker} t{t+1}: {e}")
            if t < retries - 1:
                import time as tm; tm.sleep(1)
    return None

# ── RSV: força da streak ──────────────────────────────────
def classificar_rsv_forca(dias_pos, dias_neg):
    if dias_pos >= 20: return '🔥 Muito forte', 'positivo'
    if dias_pos >= 10: return '💪 Forte',       'positivo'
    if dias_pos >= 5:  return '📈 Moderado',    'positivo'
    if dias_pos >= 1:  return '🌱 Início',      'positivo'
    if dias_neg >= 20: return '🧊 Muito fraco', 'negativo'
    if dias_neg >= 10: return '❄️ Fraco',       'negativo'
    if dias_neg >= 5:  return '📉 Mod. neg.',   'negativo'
    if dias_neg >= 1:  return '🔻 Início neg.', 'negativo'
    return '⚪ Neutro', 'neutro'

# ── Cálculo unificado por ação ────────────────────────────
def calcular_tudo(ticker, df, bova_df):
    try:
        c  = df['Close']
        h  = df['High']
        l  = df['Low']
        v  = df['Volume']

        # Alinha com BOVA11
        idx  = c.index.intersection(bova_df['Close'].index)
        c_a  = c.loc[idx];   bc_a = bova_df['Close'].loc[idx]
        v_a  = v.loc[idx];   bv_a = bova_df['Volume'].loc[idx]
        h_a  = h.loc[idx];   l_a  = l.loc[idx]

        if len(c_a) < MA_SLOW + 20:
            return None

        # ── Indicadores base ─────────────────────────────
        ma50   = c_a.rolling(MA_FAST).mean()
        ma150  = c_a.rolling(MA_MED).mean()
        ma200  = c_a.rolling(MA_SLOW).mean()
        vol_ma = v_a.rolling(20).mean()
        r5d    = h_a.rolling(CONTRACAO_JANELA).max() - l_a.rolling(CONTRACAO_JANELA).min()
        r20d   = h_a.rolling(CONTRACAO_REF).max()    - l_a.rolling(CONTRACAO_REF).min()
        max10d = h_a.rolling(BREAKOUT_JANELA).max().shift(1)
        min5d  = l_a.rolling(CONTRACAO_JANELA).min().shift(1)

        # MRS e RSV
        rp_c = c_a / bc_a
        rp_v = v_a / bv_a
        mrs  = (rp_c / rp_c.rolling(LENGTH).mean() - 1) * 100
        rsv  = (rp_v / rp_v.rolling(LENGTH).mean() - 1) * 100

        # Valores atuais
        preco   = float(c_a.iloc[-1])
        ma50_h  = float(ma50.iloc[-1])
        ma150_h = float(ma150.iloc[-1])
        ma200_h = float(ma200.iloc[-1])
        vol_h   = float(v_a.iloc[-1])
        volma_h = float(vol_ma.iloc[-1])
        r5_h    = float(r5d.iloc[-1])
        r20_h   = float(r20d.iloc[-1])
        max10_h = float(max10d.iloc[-1])
        min5_h  = float(min5d.iloc[-1])
        mrs_h   = float(mrs.iloc[-1])
        mrs_d1  = float(mrs.iloc[-2])
        rsv_h   = float(rsv.iloc[-1])

        if any(np.isnan(x) for x in [ma50_h, ma150_h, ma200_h, volma_h,
                                       r5_h, r20_h, max10_h, mrs_h, rsv_h]):
            return None

        vol_ratio  = vol_h / volma_h if volma_h > 0 else 0
        contracao  = r5_h  / r20_h  if r20_h  > 0 else 1
        dist_ma200 = (preco / ma200_h - 1) * 100 if ma200_h > 0 else 0
        breakout   = preco > max10_h
        stop       = round(min5_h, 2)
        risco_pct  = round((preco - min5_h) / preco * 100, 2) if preco > 0 else None

        ma200_20d  = float(ma200.iloc[-21]) if len(ma200) > 21 else np.nan
        ma200_sub  = (not np.isnan(ma200_20d)) and (ma200_h > ma200_20d)

        # ── RSV streak ───────────────────────────────────
        dias_rsv_pos = 0
        for i in range(1, min(60, len(rsv))):
            val = rsv.iloc[-i]
            if pd.isna(val): continue
            if float(val) > 0: dias_rsv_pos += 1
            else: break

        dias_rsv_neg = 0
        for i in range(1, min(60, len(rsv))):
            val = rsv.iloc[-i]
            if pd.isna(val): continue
            if float(val) < 0: dias_rsv_neg += 1
            else: break

        rsv_forca, rsv_lado = classificar_rsv_forca(dias_rsv_pos, dias_rsv_neg)

        # ── Sinais MRS/RSV ───────────────────────────────
        sinais = []

        if mrs_d1 <= 0 and mrs_h > 0 and rsv_h > 0:
            sinais.append({'tipo': 'COMPRA_HOJE', 'emoji': '🟢'})
        if mrs_d1 >= 0 and mrs_h < 0 and rsv_h < 0:
            sinais.append({'tipo': 'VENDA_HOJE', 'emoji': '🔴'})
        if -2 <= mrs_h < 0 and rsv_h > 0 and len(mrs) >= 3:
            if mrs.iloc[-3] < mrs.iloc[-2] < mrs_h:
                sinais.append({'tipo': 'PROXIMO_COMPRA', 'emoji': '🔶', 'distancia': round(abs(mrs_h), 2)})
        for i in range(2, min(6, len(mrs))):
            if mrs.iloc[-i-1] <= 0 and mrs.iloc[-i] > 0 and rsv.iloc[-i] > 0:
                sinais.append({'tipo': 'COMPRA_RECENTE', 'emoji': '🟢', 'dias_atras': i}); break
        for i in range(2, min(6, len(mrs))):
            if mrs.iloc[-i-1] >= 0 and mrs.iloc[-i] < 0 and rsv.iloc[-i] < 0:
                sinais.append({'tipo': 'VENDA_RECENTE', 'emoji': '🔴', 'dias_atras': i}); break
        if (mrs_h < 0 and mrs_h > -5 and rsv_h > 0 and 1 <= dias_rsv_pos <= 10):
            janela = min(10, len(mrs) - 1)
            mrs_j  = list(reversed([float(mrs.iloc[-i]) for i in range(1, janela + 1)]))
            quebras = sum(1 for i in range(len(mrs_j)-1) if mrs_j[i] >= mrs_j[i+1])
            if quebras <= 1:
                sinais.append({'tipo': 'RECUPERACAO', 'emoji': '🔼',
                               'dias_subindo': janela, 'dias_rsv_positivo': dias_rsv_pos, 'quebras': quebras})

        # ── Sinal REVERSAO MRS (whitelist) ───────────────────────
        tk_base = ticker.replace('.SA', '')
        if tk_base in REVERSAO_WHITELIST and REVERSAO_MRS_MIN <= mrs_h <= REVERSAO_MRS_MAX:
            mrs_win = mrs.iloc[-REVERSAO_OSCIL_WIN:].dropna().values
            if len(mrs_win) >= 4:
                cruzamentos = sum(
                    1 for j in range(len(mrs_win) - 1)
                    if (mrs_win[j] > 0) != (mrs_win[j+1] > 0)
                )
                if cruzamentos >= REVERSAO_MIN_CRUZ:
                    sinais.append({
                        'tipo':         'REVERSAO_MRS',
                        'emoji':        '🔁',
                        'mrs':          round(mrs_h, 2),
                        'cruzamentos':  cruzamentos,
                    })

        # ── Score APGAR ──────────────────────────────────
        scores = {}

        # C1 Tendência macro
        if preco > ma50_h > ma150_h > ma200_h and ma200_sub:
            scores['tendencia'] = 2
        elif preco > ma50_h and preco > ma200_h:
            scores['tendencia'] = 1
        else:
            scores['tendencia'] = 0

        # C2 Stage
        if preco < ma200_h:
            scores['estagio'] = 0
        elif dist_ma200 <= 25:
            scores['estagio'] = 2
        else:
            scores['estagio'] = 1

        # C3 Força relativa
        if mrs_h > 5:   scores['forca_relativa'] = 2
        elif mrs_h >= 0:scores['forca_relativa'] = 1
        else:           scores['forca_relativa'] = 0

        # C4 Contração
        if contracao < 0.35:  scores['contracao'] = 2
        elif contracao < 0.50:scores['contracao'] = 1
        else:                 scores['contracao'] = 0

        # C5 Breakout + volume
        if breakout and vol_ratio >= 2.0:     scores['breakout'] = 2
        elif breakout or vol_ratio >= 1.5:    scores['breakout'] = 1
        else:                                 scores['breakout'] = 0

        score_total = sum(scores.values())
        if score_total >= 9:   classe, cemoji = 'FORTE',    '🟢'
        elif score_total >= 7: classe, cemoji = 'POSITIVO', '🟡'
        elif score_total >= 5: classe, cemoji = 'NEUTRO',   '⚪'
        else:                  classe, cemoji = 'FRACO',    '🔴'

        return {
            'ticker':          ticker.replace('.SA', ''),
            'preco':           round(preco, 2),
            # MRS/RSV
            'mrs':             round(mrs_h, 2),
            'rsv':             round(rsv_h, 2),
            'dias_rsv_positivo': int(dias_rsv_pos),
            'dias_rsv_negativo': int(dias_rsv_neg),
            'rsv_forca':       rsv_forca,
            'rsv_lado':        rsv_lado,
            'sinais':          sinais,
            # APGAR
            'score':           score_total,
            'classe':          classe,
            'emoji':           cemoji,
            'scores':          scores,
            'ma50':            round(ma50_h, 2),
            'ma150':           round(ma150_h, 2),
            'ma200':           round(ma200_h, 2),
            'ma200_subindo':   ma200_sub,
            'vol_ratio':       round(vol_ratio, 2),
            'contracao':       round(contracao, 2),
            'breakout':        breakout,
            'stop':            stop,
            'risco_pct':       risco_pct,
            'dist_ma200':      round(dist_ma200, 1),
        }

    except Exception as e:
        print(f"[ERR] {ticker}: {e}")
        return None

# ── Processamento principal ───────────────────────────────
def processar_screener():
    print("[INFO] Iniciando screener unificado...")

    bova = baixar('BOVA11.SA')
    if bova is None:
        return None

    todas = []
    for i, ticker in enumerate(ACOES_PRINCIPAIS, 1):
        print(f"[INFO] {i}/{len(ACOES_PRINCIPAIS)}: {ticker}")
        df = baixar(ticker)
        if df is None:
            continue
        res = calcular_tudo(ticker, df, bova)
        if res:
            todas.append(res)

    # ── MRS/RSV: monta listas de sinais ──────────────────
    sinais_hoje, proximos, recentes, recuperacoes, reversao_mrs = [], [], [], [], []
    for a in todas:
        base = {k: a[k] for k in ['ticker','mrs','rsv','preco',
                                    'dias_rsv_positivo','dias_rsv_negativo',
                                    'rsv_forca','rsv_lado']}
        for s in a['sinais']:
            item = {**base}
            if s['tipo'] in ('COMPRA_HOJE','VENDA_HOJE'):
                item['tipo']  = 'COMPRA' if 'COMPRA' in s['tipo'] else 'VENDA'
                item['emoji'] = s['emoji']
                sinais_hoje.append(item)
            elif s['tipo'] == 'PROXIMO_COMPRA':
                item['distancia'] = s['distancia']
                proximos.append(item)
            elif s['tipo'] in ('COMPRA_RECENTE','VENDA_RECENTE'):
                item['tipo']     = 'COMPRA' if 'COMPRA' in s['tipo'] else 'VENDA'
                item['diasAtras']= s['dias_atras']
                recentes.append(item)
            elif s['tipo'] == 'RECUPERACAO':
                item['dias_subindo']      = s['dias_subindo']
                item['dias_rsv_positivo'] = s['dias_rsv_positivo']
                item['quebras']           = s['quebras']
                recuperacoes.append(item)
            elif s['tipo'] == 'REVERSAO_MRS':
                item['mrs']        = s['mrs']
                item['cruzamentos']= s['cruzamentos']
                reversao_mrs.append(item)

    top_mrs = sorted(todas, key=lambda x: x['mrs'], reverse=True)[:10]
    top_rsv_pos = sorted([a for a in todas if a['dias_rsv_positivo'] > 0],
                          key=lambda x: x['dias_rsv_positivo'], reverse=True)[:10]
    top_rsv_neg = sorted([a for a in todas if a['dias_rsv_negativo'] > 0],
                          key=lambda x: x['dias_rsv_negativo'], reverse=True)[:10]
    recuperacoes.sort(key=lambda x: (x['dias_rsv_positivo'], x['dias_subindo']), reverse=True)

    # ── APGAR: ordenado por score ─────────────────────────
    apgar_lista = sorted(todas, key=lambda x: (x['score'], x['mrs']), reverse=True)

    # ── Sinais VCP (APGAR) ──────────────────────────────────
    vcp_sinais = [
        {'ticker': a['ticker'], 'tipo': 'VCP', 'preco': a['preco']}
        for a in apgar_lista
        if 5 <= a['score'] <= 8
        and a['scores'].get('breakout') == 2
        and a['mrs'] > 0
        and a['rsv'] > 0
    ]

    agora_br = datetime.utcnow() - timedelta(hours=3)
    return {
        'lastUpdate': agora_br.strftime('%d/%m/%Y %H:%M:%S'),
        'dataDados':  bova.index[-1].strftime('%d/%m/%Y'),
        'totalAcoes': len(ACOES_PRINCIPAIS),
        # MRS/RSV
        'sinaisHoje':         sinais_hoje,
        'sinaisCompraHoje':   [
            {'ticker': s['ticker'], 'tipo': 'MRS',     'preco': s.get('preco')}
            for s in sinais_hoje if s.get('tipo') == 'COMPRA'
        ] + [
            {'ticker': a['ticker'], 'tipo': 'VCP',     'preco': a.get('preco')}
            for a in vcp_sinais
        ] + [
            {'ticker': r['ticker'], 'tipo': 'REVERSAO', 'preco': r.get('preco')}
            for r in reversao_mrs
        ],
        'proximosCruzar':     proximos,
        'cruzamentosRecentes':recentes,
        'recuperacoes':       recuperacoes,
        'topMRS':             [{k: a[k] for k in ['ticker','mrs','rsv','preco','dias_rsv_positivo','dias_rsv_negativo','rsv_forca','rsv_lado']} for a in top_mrs],
        'topRSVConsistente':  [{k: a[k] for k in ['ticker','mrs','rsv','preco','dias_rsv_positivo','rsv_forca']} for a in top_rsv_pos],
        'topRSVFraco':        [{k: a[k] for k in ['ticker','mrs','rsv','preco','dias_rsv_negativo','rsv_forca']} for a in top_rsv_neg],
        # APGAR
        'apgar': {
            'acoes': apgar_lista,
            'resumo': {
                'forte':    sum(1 for a in apgar_lista if a['score'] >= 9),
                'positivo': sum(1 for a in apgar_lista if 7 <= a['score'] < 9),
                'neutro':   sum(1 for a in apgar_lista if 5 <= a['score'] < 7),
                'fraco':    sum(1 for a in apgar_lista if a['score'] < 5),
            }
        },
        'reversaoMRS':        reversao_mrs,
        'cacheInfo': {'cached': False, 'dataProcessamento': data_pregao().isoformat()}
    }

# ── Handler HTTP ─────────────────────────────────────────
class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.end_headers()
        try:
            if cache_valido():
                r = _cache['data'].copy(); r['cacheInfo']['cached'] = True
                self.wfile.write(json.dumps(r).encode()); return
            if _cache['processando']:
                if _cache['data']:
                    r = _cache['data'].copy(); r['cacheInfo']['cached'] = True
                    self.wfile.write(json.dumps(r).encode())
                else:
                    self.wfile.write(json.dumps({'error':'Processando, tente em 30s'}).encode())
                return
            _cache['processando'] = True
            r = processar_screener()
            if r:
                _cache['data'] = r; _cache['data_proc'] = data_pregao()
                self.wfile.write(json.dumps(r).encode())
            else:
                self.wfile.write(json.dumps({'error':'Dados indisponíveis','retry':True}).encode())
        except Exception as e:
            import traceback
            self.wfile.write(json.dumps({'error':str(e),'trace':traceback.format_exc()}).encode())
        finally:
            _cache['processando'] = False

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin','*')
        self.send_header('Access-Control-Allow-Methods','GET, OPTIONS')
        self.end_headers()