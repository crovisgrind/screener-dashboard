# api/screener.py
# API com cache diário - atualiza apenas 1x por dia após fechamento da B3

from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf
import pandas as pd
from datetime import datetime, time, timedelta
import os

# Lista de ações
ACOES_PRINCIPAIS = [
    "ALOS3.SA","ABEV3.SA","ANIM3.SA","ASAI3.SA","AURE3.SA","AXIA3.SA","AXIA6.SA","AXIA7.SA","AZZA3.SA","B3SA3.SA","BBSE3.SA","BBDC3.SA","BBDC4.SA","BRAP4.SA","BBAS3.SA","BRKM5.SA","BRAV3.SA","BPAC11.SA","CXSE3.SA","BHIA3.SA","CBAV3.SA","CEAB3.SA","CMIG4.SA","COGN3.SA","CSMG3.SA","CPLE3.SA","CSAN3.SA","CPFE3.SA","CMIN3.SA","CURY3.SA","CVCB3.SA","CYRE3.SA","CYRE4.SA","DIRR3.SA","ECOR3.SA","EMBJ3.SA","ENGI11.SA","ENEV3.SA","EGIE3.SA","EQTL3.SA","EZTC3.SA","FLRY3.SA","GGBR4.SA","GOAU4.SA","GGPS3.SA","GMAT3.SA","HAPV3.SA","HYPE3.SA","IGTI11.SA","INTB3.SA","IRBR3.SA","ISAE4.SA","ITSA4.SA","ITUB4.SA","KLBN11.SA","RENT3.SA","RENT4.SA","LREN3.SA","LWSA3.SA","MGLU3.SA","POMO4.SA","MBRF3.SA","BEEF3.SA","MOTV3.SA","MOVI3.SA","MRVE3.SA","MULT3.SA","NATU3.SA","NEOE3.SA","PCAR3.SA","PETR3.SA","PETR4.SA","RECV3.SA","PRIO3.SA","AUAU3.SA","PSSA3.SA","RADL3.SA","RAIZ4.SA","RAPT4.SA","RDOR3.SA","RAIL3.SA","SBSP3.SA","SAPR11.SA","SANB11.SA","SMTO3.SA","CSNA3.SA","SIMH3.SA","SLCE3.SA","SMFT3.SA","SUZB3.SA","TAEE11.SA","VIVT3.SA","TEND3.SA","TIMS3.SA","TOTS3.SA","UGPA3.SA","USIM5.SA","VALE3.SA","VAMO3.SA","VBBR3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA"
]

LENGTH = 200

# ==================== CACHE DIÁRIO ====================
_cache_diario = {
    'data': None,
    'data_processamento': None,
    'em_processamento': False
}

def obter_data_pregao_atual():
    """Retorna a data do pregão atual (considerando timezone BR = UTC-3)"""
    agora_utc = datetime.utcnow()
    agora_br = agora_utc - timedelta(hours=3)
    hora_limite = time(18, 30)
    if agora_br.time() < hora_limite:
        data_pregao = (agora_br - timedelta(days=1)).date()
    else:
        data_pregao = agora_br.date()
    return data_pregao

def cache_valido():
    """Verifica se o cache ainda é válido para o pregão de hoje"""
    data_pregao_atual = obter_data_pregao_atual()
    if _cache_diario['data'] is None:
        return False
    if _cache_diario['data_processamento'] != data_pregao_atual:
        return False
    return True

# ==================== FUNÇÕES DE PROCESSAMENTO ====================

def baixar_dados(ticker, max_retries=2):
    """Baixa dados com retry"""
    for tentativa in range(max_retries):
        try:
            print(f"[INFO] Baixando {ticker} (tentativa {tentativa + 1}/{max_retries})...")
            df = yf.download(
                ticker,
                period='1y',
                interval='1d',
                progress=False,
                auto_adjust=True,
                prepost=False,
                actions=False,
                keepna=False,
                threads=False
            )
            if df is None or len(df) == 0:
                print(f"[WARN] {ticker}: DataFrame vazio na tentativa {tentativa + 1}")
                if tentativa < max_retries - 1:
                    continue
                return None

            if hasattr(df, 'columns'):
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"[ERROR] {ticker}: Faltam colunas {missing_cols}")
                return None

            if len(df) < LENGTH:
                print(f"[WARN] {ticker}: Apenas {len(df)} dias (mínimo {LENGTH})")
                return None

            print(f"[SUCCESS] {ticker}: {len(df)} dias baixados")
            return df

        except Exception as e:
            print(f"[ERROR] Tentativa {tentativa + 1} falhou para {ticker}: {str(e)}")
            if tentativa < max_retries - 1:
                import time
                time.sleep(1)
                continue
            return None
    return None


def classificar_rsv_forca(dias_positivo, dias_negativo):
    """
    Classifica a força/fraqueza do RSV com granularidade nos dois lados.
    Retorna (emoji + label, lado) onde lado é 'positivo' ou 'negativo'.
    """
    if dias_positivo >= 1:
        # Lado positivo — volume sustentado acima do mercado
        if dias_positivo >= 20:
            return '🔥 Muito forte', 'positivo'
        elif dias_positivo >= 10:
            return '💪 Forte', 'positivo'
        elif dias_positivo >= 5:
            return '📈 Moderado', 'positivo'
        else:
            return '🌱 Início', 'positivo'
    elif dias_negativo >= 1:
        # Lado negativo — volume sustentado abaixo do mercado
        if dias_negativo >= 20:
            return '🧊 Muito fraco', 'negativo'
        elif dias_negativo >= 10:
            return '❄️ Fraco', 'negativo'
        elif dias_negativo >= 5:
            return '📉 Moderado neg.', 'negativo'
        else:
            return '🔻 Início neg.', 'negativo'
    else:
        return '⚪ Neutro', 'neutro'


def calcular_sinais(ticker, bova_data):
    """Calcula sinais para uma ação"""
    try:
        df = baixar_dados(ticker)
        if df is None or len(df) < LENGTH:
            return None

        df_aligned, bova_aligned = df.align(bova_data, join='inner')
        if len(df_aligned) < LENGTH:
            return None

        # MRS
        rp = df_aligned['Close'] / bova_aligned['Close']
        mrs = (rp / rp.rolling(LENGTH).mean() - 1) * 100

        # RSV
        rp_vol = df_aligned['Volume'] / bova_aligned['Volume']
        rsv = (rp_vol / rp_vol.rolling(LENGTH).mean() - 1) * 100

        if len(mrs) < 3:
            return None

        ticker_limpo = ticker.replace('.SA', '')
        preco = float(df_aligned['Close'].iloc[-1])
        mrs_atual = float(mrs.iloc[-1])
        rsv_atual = float(rsv.iloc[-1])
        mrs_ontem = float(mrs.iloc[-2])

        # ============ CONSISTÊNCIA DO RSV ============
        dias_rsv_positivo = 0
        for i in range(1, min(60, len(rsv))):
            val = rsv.iloc[-i]
            if pd.isna(val):
                continue
            if float(val) > 0:
                dias_rsv_positivo += 1
            else:
                break

        dias_rsv_negativo = 0
        for i in range(1, min(60, len(rsv))):
            val = rsv.iloc[-i]
            if pd.isna(val):
                continue
            if float(val) < 0:
                dias_rsv_negativo += 1
            else:
                break

        # Usa a nova função com granularidade nos dois lados
        rsv_forca, rsv_lado = classificar_rsv_forca(dias_rsv_positivo, dias_rsv_negativo)

        print(f"[DEBUG] {ticker_limpo}: dias_rsv_pos={dias_rsv_positivo}, dias_rsv_neg={dias_rsv_negativo}, forca={rsv_forca}")

        resultado = {
            'ticker': ticker_limpo,
            'preco': round(preco, 2),
            'mrs': round(mrs_atual, 2),
            'rsv': round(rsv_atual, 2),
            'dias_rsv_positivo': int(dias_rsv_positivo),
            'dias_rsv_negativo': int(dias_rsv_negativo),
            'rsv_forca': rsv_forca,
            'rsv_lado': rsv_lado,
            'sinais': []
        }

        # Sinal de COMPRA HOJE
        if mrs_ontem <= 0 and mrs_atual > 0 and rsv_atual > 0:
            resultado['sinais'].append({'tipo': 'COMPRA_HOJE', 'emoji': '🟢'})

        # Sinal de VENDA HOJE
        if mrs_ontem >= 0 and mrs_atual < 0 and rsv_atual < 0:
            resultado['sinais'].append({'tipo': 'VENDA_HOJE', 'emoji': '🔴'})

        # Próximo de cruzar
        if -2 <= mrs_atual < 0 and rsv_atual > 0:
            if len(mrs) >= 3 and mrs.iloc[-3] < mrs.iloc[-2] < mrs_atual:
                resultado['sinais'].append({
                    'tipo': 'PROXIMO_COMPRA',
                    'emoji': '🔶',
                    'distancia': round(abs(mrs_atual), 2)
                })

        # Cruzamentos recentes
        for i in range(2, min(6, len(mrs))):
            if mrs.iloc[-i-1] <= 0 and mrs.iloc[-i] > 0 and rsv.iloc[-i] > 0:
                resultado['sinais'].append({
                    'tipo': 'COMPRA_RECENTE',
                    'emoji': '🟢',
                    'dias_atras': i
                })
                break

        for i in range(2, min(6, len(mrs))):
            if mrs.iloc[-i-1] >= 0 and mrs.iloc[-i] < 0 and rsv.iloc[-i] < 0:
                resultado['sinais'].append({
                    'tipo': 'VENDA_RECENTE',
                    'emoji': '🔴',
                    'dias_atras': i
                })
                break

        # Sinal amarelo
        if (mrs_atual < 0 and len(mrs) >= 3 and
                mrs.iloc[-3] < mrs.iloc[-2] < mrs_atual and
                rsv_atual > 0 and rsv.iloc[-2] > 0 and rsv.iloc[-3] > 0):
            resultado['sinais'].append({'tipo': 'ATENCAO', 'emoji': '🟡'})

        return resultado

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")
        return None


def processar_screener():
    """Processa o screener completo"""
    print(f"[INFO] Iniciando processamento do screener...")

    bova_data = baixar_dados('BOVA11.SA')
    if bova_data is None:
        print("[ERROR] BOVA11.SA indisponível")
        return None

    print(f"[INFO] BOVA11 baixado com sucesso: {len(bova_data)} dias")

    todas_acoes = []
    sinais_hoje = []
    proximos_cruzar = []
    cruzamentos_recentes = []

    for i, ticker in enumerate(ACOES_PRINCIPAIS, 1):
        print(f"[INFO] Processando {i}/{len(ACOES_PRINCIPAIS)}: {ticker}")
        resultado = calcular_sinais(ticker, bova_data)

        if resultado:
            todas_acoes.append(resultado)

            for sinal in resultado['sinais']:
                # Campos comuns a todos os sinais — agora inclui RSV consistência
                item = {
                    'ticker': resultado['ticker'],
                    'mrs': resultado['mrs'],
                    'rsv': resultado['rsv'],
                    'preco': resultado['preco'],
                    'dias_rsv_positivo': resultado['dias_rsv_positivo'],
                    'dias_rsv_negativo': resultado['dias_rsv_negativo'],
                    'rsv_forca': resultado['rsv_forca'],
                    'rsv_lado': resultado['rsv_lado'],
                }

                if sinal['tipo'] in ['COMPRA_HOJE', 'VENDA_HOJE']:
                    item['tipo'] = 'COMPRA' if 'COMPRA' in sinal['tipo'] else 'VENDA'
                    item['emoji'] = sinal['emoji']
                    sinais_hoje.append(item)

                elif sinal['tipo'] == 'PROXIMO_COMPRA':
                    item['distancia'] = sinal['distancia']
                    proximos_cruzar.append(item)

                elif sinal['tipo'] in ['COMPRA_RECENTE', 'VENDA_RECENTE']:
                    item['tipo'] = 'COMPRA' if 'COMPRA' in sinal['tipo'] else 'VENDA'
                    item['diasAtras'] = sinal['dias_atras']
                    cruzamentos_recentes.append(item)

    # ==================== TOP MRS ====================
    todas_acoes_ordenadas = sorted(todas_acoes, key=lambda x: x['mrs'], reverse=True)
    top_mrs = [
        {
            'ticker': a['ticker'],
            'mrs': a['mrs'],
            'rsv': a['rsv'],
            'preco': a['preco'],
            'dias_rsv_positivo': a['dias_rsv_positivo'],
            'dias_rsv_negativo': a['dias_rsv_negativo'],
            'rsv_forca': a['rsv_forca'],
            'rsv_lado': a['rsv_lado'],
        }
        for a in todas_acoes_ordenadas[:10]
    ]

    # ==================== TOP RSV CONSISTENTE (POSITIVO) ====================
    # Ações que sustentam volume acima do mercado há mais dias consecutivos
    acoes_rsv_positivo = [a for a in todas_acoes if a['dias_rsv_positivo'] > 0]
    acoes_rsv_positivo_ord = sorted(acoes_rsv_positivo, key=lambda x: x['dias_rsv_positivo'], reverse=True)
    top_rsv_consistente = [
        {
            'ticker': a['ticker'],
            'mrs': a['mrs'],
            'rsv': a['rsv'],
            'preco': a['preco'],
            'dias_rsv_positivo': a['dias_rsv_positivo'],
            'rsv_forca': a['rsv_forca'],
        }
        for a in acoes_rsv_positivo_ord[:10]
    ]

    # ==================== TOP RSV FRACO (NEGATIVO) ====================
    # Ações que sustentam volume abaixo do mercado há mais dias consecutivos
    acoes_rsv_negativo = [a for a in todas_acoes if a['dias_rsv_negativo'] > 0]
    acoes_rsv_negativo_ord = sorted(acoes_rsv_negativo, key=lambda x: x['dias_rsv_negativo'], reverse=True)
    top_rsv_fraco = [
        {
            'ticker': a['ticker'],
            'mrs': a['mrs'],
            'rsv': a['rsv'],
            'preco': a['preco'],
            'dias_rsv_negativo': a['dias_rsv_negativo'],
            'rsv_forca': a['rsv_forca'],
        }
        for a in acoes_rsv_negativo_ord[:10]
    ]

    # Obter última data de dados
    ultima_data = bova_data.index[-1].strftime('%d/%m/%Y')

    agora_utc = datetime.utcnow()
    agora_br = agora_utc - timedelta(hours=3)

    resposta = {
        'lastUpdate': agora_br.strftime('%d/%m/%Y %H:%M:%S'),
        'dataDados': ultima_data,
        'timestamp': int(agora_utc.timestamp()),
        'totalAcoes': len(ACOES_PRINCIPAIS),
        'sinaisHoje': sinais_hoje,
        'proximosCruzar': proximos_cruzar,
        'cruzamentosRecentes': cruzamentos_recentes,
        'topMRS': top_mrs,
        'topRSVConsistente': top_rsv_consistente,   # ✅ NOVO — maiores RSV positivos consecutivos
        'topRSVFraco': top_rsv_fraco,               # ✅ NOVO — maiores RSV negativos consecutivos
        'cacheInfo': {
            'cached': False,
            'dataProcessamento': obter_data_pregao_atual().isoformat()
        }
    }

    print(f"[INFO] Processamento concluído: {len(sinais_hoje)} sinais hoje")
    return resposta


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'public, max-age=3600')
        self.end_headers()

        try:
            if cache_valido():
                print("[INFO] Usando cache do dia")
                resposta = _cache_diario['data'].copy()
                resposta['cacheInfo']['cached'] = True
                self.wfile.write(json.dumps(resposta).encode())
                return

            if _cache_diario['em_processamento']:
                print("[INFO] Processamento em andamento, aguardando...")
                if _cache_diario['data']:
                    resposta = _cache_diario['data'].copy()
                    resposta['cacheInfo']['cached'] = True
                    resposta['cacheInfo']['processando'] = True
                    self.wfile.write(json.dumps(resposta).encode())
                else:
                    self.wfile.write(json.dumps({
                        'error': 'Processamento em andamento, tente novamente em 30 segundos'
                    }).encode())
                return

            _cache_diario['em_processamento'] = True
            print("[INFO] Cache inválido, processando nova análise...")

            resposta = processar_screener()

            if resposta:
                _cache_diario['data'] = resposta
                _cache_diario['data_processamento'] = obter_data_pregao_atual()
                self.wfile.write(json.dumps(resposta).encode())
            else:
                self.wfile.write(json.dumps({
                    'error': 'Dados temporariamente indisponíveis',
                    'message': 'O Yahoo Finance está com problemas no BOVA11. Tente novamente em alguns minutos.',
                    'retry': True,
                    'timestamp': datetime.utcnow().isoformat()
                }).encode())

        except Exception as e:
            print(f"[ERROR] Erro crítico: {e}")
            import traceback
            error_details = traceback.format_exc()
            print(error_details)
            self.wfile.write(json.dumps({
                'error': 'Erro ao processar screener',
                'details': str(e),
                'trace': error_details if 'VERCEL_ENV' not in os.environ else None
            }).encode())

        finally:
            _cache_diario['em_processamento'] = False

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()