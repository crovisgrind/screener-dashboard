# api/screener.py
# API Serverless para rodar no Vercel

from http.server import BaseHTTPRequestHandler
import json
import yfinance as yf
import pandas as pd
from datetime import datetime

# Lista de ações (pode ser todas as 103 ou um subconjunto)
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
"USIM5.SA","VALE3.SA","VAMO3.SA","VBBR3.SA","VIVA3.SA","WEGE3.SA","YDUQ3.SA"					
]

LENGTH = 200

def baixar_dados(ticker):
    try:
        df = yf.download(ticker, period='1y', interval='1d', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df if len(df) > LENGTH else None
    except:
        return None

def calcular_sinais(ticker, bova_data):
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
        
        resultado = {
            'ticker': ticker_limpo,
            'preco': round(preco, 2),
            'mrs': round(mrs_atual, 2),
            'rsv': round(rsv_atual, 2),
            'sinais': []
        }
        
        # Sinal de COMPRA HOJE
        if mrs_ontem <= 0 and mrs_atual > 0 and rsv_atual > 0:
            resultado['sinais'].append({
                'tipo': 'COMPRA_HOJE',
                'emoji': '🟢'
            })
        
        # Sinal de VENDA HOJE
        if mrs_ontem >= 0 and mrs_atual < 0 and rsv_atual < 0:
            resultado['sinais'].append({
                'tipo': 'VENDA_HOJE',
                'emoji': '🔴'
            })
        
        # Próximo de cruzar para CIMA
        if -2 <= mrs_atual < 0 and rsv_atual > 0:
            if len(mrs) >= 3 and mrs.iloc[-3] < mrs.iloc[-2] < mrs_atual:
                resultado['sinais'].append({
                    'tipo': 'PROXIMO_COMPRA',
                    'emoji': '🔶',
                    'distancia': round(abs(mrs_atual), 2)
                })
        
        # Cruzamentos recentes (últimos 5 dias)
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
            resultado['sinais'].append({
                'tipo': 'ATENCAO',
                'emoji': '🟡'
            })
        
        return resultado
        
    except Exception as e:
        print(f"Erro em {ticker}: {e}")
        return None

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        # Headers CORS + Desabilitar Cache
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        self.end_headers()
        
        try:
            # Baixar BOVA11
            bova_data = baixar_dados('BOVA11.SA')
            
            if bova_data is None:
                self.wfile.write(json.dumps({
                    'error': 'Erro ao baixar dados do BOVA11'
                }).encode())
                return
            
            # Processar ações
            todas_acoes = []
            sinais_hoje = []
            proximos_cruzar = []
            cruzamentos_recentes = []
            
            for ticker in ACOES_PRINCIPAIS:
                resultado = calcular_sinais(ticker, bova_data)
                
                if resultado:
                    todas_acoes.append(resultado)
                    
                    # Classificar sinais
                    for sinal in resultado['sinais']:
                        item = {
                            'ticker': resultado['ticker'],
                            'mrs': resultado['mrs'],
                            'rsv': resultado['rsv'],
                            'preco': resultado['preco']
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
            
            # Top MRS
            todas_acoes_ordenadas = sorted(todas_acoes, key=lambda x: x['mrs'], reverse=True)
            top_mrs = [
                {
                    'ticker': acao['ticker'],
                    'mrs': acao['mrs'],
                    'preco': acao['preco']
                }
                for acao in todas_acoes_ordenadas[:10]
            ]
            
            # Montar resposta
            resposta = {
                'lastUpdate': datetime.now().strftime('%d/%m/%Y %H:%M:%S'),
                'totalAcoes': len(ACOES_PRINCIPAIS),
                'sinaisHoje': sinais_hoje,
                'proximosCruzar': proximos_cruzar,
                'cruzamentosRecentes': cruzamentos_recentes,
                'topMRS': top_mrs
            }
            
            self.wfile.write(json.dumps(resposta).encode())
            
        except Exception as e:
            self.wfile.write(json.dumps({
                'error': str(e)
            }).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()