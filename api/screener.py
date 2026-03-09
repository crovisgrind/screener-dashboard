# api/screener.py
# Screener VCP (Qullamaggie) para Ibovespa
# Três estados por ação: SINAL (VCP completo), CONTRACAO (aguardando breakout), WATCHLIST (tendência ok)

from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, time, timedelta
import os

ACOES_PRINCIPAIS = [
    "ALOS3.SA","ABEV3.SA","ANIM3.SA","ASAI3.SA","AURE3.SA","AXIA3.SA","AXIA6.SA","AXIA7.SA","AZZA3.SA","B3SA3.SA","BBSE3.SA","BBDC3.SA","BBDC4.SA","BRAP4.SA","BBAS3.SA","BRKM5.SA","BRAV3.SA","BPAC11.SA","CXSE3.SA","BHIA3.SA","CBAV3.SA","CEAB3.SA","CMIG4.SA","COGN3.SA","CSMG3.SA","CPLE3.SA","CSAN3.SA","CPFE3.SA","CMIN3.SA","CURY3.SA","CVCB3.SA","CYRE3.SA","CYRE4.SA","DIRR3.SA","ECOR3.SA","EMBJ3.SA","ENGI11.SA","ENEV3.SA","EGIE3.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GGPS3.SA","GMAT3.SA","HAPV3.SA","HYPE3.SA","IGTI11.SA","INTB3.SA","IRBR3.SA","ISAE4.SA","ITSA4.SA","ITUB4.SA","KLBN11.SA","RENT3.SA","RENT4.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","POMO4.SA","MBRF3.SA","BEEF3.SA","MOTV3.SA","MOVI3.SA","MRVE3.SA","MULT3.SA","NATU3.SA","NEOE3.SA","PCAR3.SA","PETR3.SA","PETR4.SA","RECV3.SA","PRIO3.SA","AUAU3.SA","PSSA3.SA","RADL3.SA","RAIZ4.SA","RAPT4.SA","RDOR3.SA","RAIL3.SA","SBSP3.SA","SAPR11.SA","SANB11.SA","SMTO3.SA","CSNA3.SA","SIMH3.SA","SLCE3.SA","SMFT3.SA","SUZB3.SA","TAEE11.SA","VIVT3.SA","TEND3.SA","TIMS3.SA","TOTS3.SA","UGPA3.SA","USIM5.SA","VALE3.SA","VAMO3.SA","VBBR3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA"
]

# ── Parâmetros VCP ──────────────────────────────────────
PRECO_MIN        = 5.00
MA_FAST          = 50
MA_SLOW          = 200
VOL_MEDIO_MIN    = 500_000
CONTRACAO_JANELA = 5
CONTRACAO_REF    = 20
CONTRACAO_FATOR  = 0.50   # range5d / range20d < este valor
BREAKOUT_JANELA  = 10
VOLUME_FATOR     = 2.0    # vol hoje > X * vol_ma20

# ── Cache ────────────────────────────────────────────────
_cache_diario = {'data': None, 'data_processamento': None, 'em_processamento': False}

def obter_data_pregao_atual():
    agora_br = datetime.utcnow() - timedelta(hours=3)
    if agora_br.time() < time(18, 30):
        return (agora_br - timedelta(days=1)).date()
    return agora_br.date()

def cache_valido():
    if _cache_diario['data'] is None:
        return False
    return _cache_diario['data_processamento'] == obter_data_pregao_atual()

# ── Download ─────────────────────────────────────────────
def baixar_dados(ticker, max_retries=2):
    for tentativa in range(max_retries):
        try:
            df = yf.download(ticker, period='2y', interval='1d',
                             progress=False, auto_adjust=True,
                             prepost=False, actions=False, threads=False)
            if df is None or len(df) == 0:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            required = ['Open','High','Low','Close','Volume']
            if any(c not in df.columns for c in required):
                return None
            if len(df) < MA_SLOW + CONTRACAO_REF + 5:
                return None
            return df
        except Exception as e:
            print(f"[ERROR] {ticker} tentativa {tentativa+1}: {e}")
            if tentativa < max_retries - 1:
                import time as t; t.sleep(1)
    return None

# ── Calcular VCP ─────────────────────────────────────────
def calcular_vcp(ticker, df):
    try:
        c   = df['Close']
        h   = df['High']
        l   = df['Low']
        v   = df['Volume']

        ma50   = c.rolling(MA_FAST).mean()
        ma200  = c.rolling(MA_SLOW).mean()
        vol_ma = v.rolling(20).mean()
        r5d    = h.rolling(CONTRACAO_JANELA).max() - l.rolling(CONTRACAO_JANELA).min()
        r20d   = h.rolling(CONTRACAO_REF).max()    - l.rolling(CONTRACAO_REF).min()
        max10d = h.rolling(BREAKOUT_JANELA).max().shift(1)
        min5d  = l.rolling(CONTRACAO_JANELA).min().shift(1)

        # Valores atuais
        preco    = float(c.iloc[-1])
        ma50_h   = float(ma50.iloc[-1])
        ma200_h  = float(ma200.iloc[-1])
        vol_h    = float(v.iloc[-1])
        volma_h  = float(vol_ma.iloc[-1])
        r5_h     = float(r5d.iloc[-1])
        r20_h    = float(r20d.iloc[-1])
        max10_h  = float(max10d.iloc[-1])
        min5_h   = float(min5d.iloc[-1])

        if any(np.isnan(x) for x in [ma50_h, ma200_h, volma_h, r5_h, r20_h, max10_h, min5_h]):
            return None

        vol_ratio    = vol_h / volma_h if volma_h > 0 else 0
        contracao    = r5_h / r20_h   if r20_h  > 0 else 1
        stop         = round(min5_h, 2)
        risco_pct    = round((preco - min5_h) / preco * 100, 2) if preco > 0 else None

        resultado = {
            'ticker':      ticker.replace('.SA',''),
            'preco':       round(preco, 2),
            'ma50':        round(ma50_h, 2),
            'ma200':       round(ma200_h, 2),
            'acima_ma50':  preco > ma50_h,
            'acima_ma200': preco > ma200_h,
            'vol_ratio':   round(vol_ratio, 2),
            'contracao':   round(contracao, 2),
            'stop':        stop,
            'risco_pct':   risco_pct,
            'estado':      None,   # SINAL | CONTRACAO | WATCHLIST | None
        }

        # Filtros base obrigatórios
        tendencia_ok = preco > ma50_h and preco > ma200_h
        preco_ok     = preco >= PRECO_MIN
        liquidez_ok  = volma_h >= VOL_MEDIO_MIN

        if not (tendencia_ok and preco_ok and liquidez_ok):
            return resultado   # retorna sem estado — não aparece em nenhuma lista

        em_contracao = contracao < CONTRACAO_FATOR

        if em_contracao:
            breakout = preco > max10_h
            vol_ok   = vol_ratio >= VOLUME_FATOR

            if breakout and vol_ok:
                resultado['estado'] = 'SINAL'
            else:
                resultado['estado'] = 'CONTRACAO'
                resultado['falta_breakout'] = not breakout
                resultado['falta_volume']   = not vol_ok
        else:
            resultado['estado'] = 'WATCHLIST'

        return resultado

    except Exception as e:
        print(f"[ERROR] {ticker} VCP: {e}")
        return None

# ── Processamento principal ───────────────────────────────
def processar_screener():
    print("[INFO] Iniciando processamento VCP...")

    sinais      = []   # VCP completo — entrar amanhã
    contracoes  = []   # Comprimindo — monitorar
    watchlist   = []   # Tendência ok mas sem contração

    for i, ticker in enumerate(ACOES_PRINCIPAIS, 1):
        print(f"[INFO] {i}/{len(ACOES_PRINCIPAIS)}: {ticker}")
        df = baixar_dados(ticker)
        if df is None:
            continue
        res = calcular_vcp(ticker, df)
        if res is None or res['estado'] is None:
            continue

        if res['estado'] == 'SINAL':
            sinais.append(res)
        elif res['estado'] == 'CONTRACAO':
            contracoes.append(res)
        elif res['estado'] == 'WATCHLIST':
            watchlist.append(res)

    # Ordena contrações pela mais comprimida primeiro
    contracoes.sort(key=lambda x: x['contracao'])
    # Ordena sinais pelo maior volume ratio
    sinais.sort(key=lambda x: x['vol_ratio'], reverse=True)
    # Watchlist: ordena por contração (as mais próximas de comprimir primeiro)
    watchlist.sort(key=lambda x: x['contracao'])

    agora_br = datetime.utcnow() - timedelta(hours=3)

    return {
        'lastUpdate':   agora_br.strftime('%d/%m/%Y %H:%M:%S'),
        'totalAcoes':   len(ACOES_PRINCIPAIS),
        'sinais':       sinais,
        'contracoes':   contracoes[:20],
        'watchlist':    watchlist[:20],
        'cacheInfo': {
            'cached': False,
            'dataProcessamento': obter_data_pregao_atual().isoformat()
        }
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
                resposta = _cache_diario['data'].copy()
                resposta['cacheInfo']['cached'] = True
                self.wfile.write(json.dumps(resposta).encode())
                return

            if _cache_diario['em_processamento']:
                if _cache_diario['data']:
                    resposta = _cache_diario['data'].copy()
                    resposta['cacheInfo']['cached'] = True
                    self.wfile.write(json.dumps(resposta).encode())
                else:
                    self.wfile.write(json.dumps({'error': 'Processando, tente em 30s'}).encode())
                return

            _cache_diario['em_processamento'] = True
            resposta = processar_screener()

            if resposta:
                _cache_diario['data'] = resposta
                _cache_diario['data_processamento'] = obter_data_pregao_atual()
                self.wfile.write(json.dumps(resposta).encode())
            else:
                self.wfile.write(json.dumps({'error': 'Dados indisponíveis', 'retry': True}).encode())

        except Exception as e:
            import traceback
            self.wfile.write(json.dumps({'error': str(e), 'trace': traceback.format_exc()}).encode())
        finally:
            _cache_diario['em_processamento'] = False

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.end_headers()