import yfinance as yf
import pandas as pd
import numpy as np

START = "2018-01-01"
CAPITAL_INICIAL = 50_000
RISK_PER_TRADE = 0.05
MAX_POSICOES = 3

TICKERS = [
    "ALOS3.SA", "ABEV3.SA", "ASAI3.SA", "AURE3.SA", "AZZA3.SA", "B3SA3.SA", "BBSE3.SA",
    "BBDC3.SA", "BBDC4.SA", "BRAP4.SA", "BBAS3.SA", "BRKM5.SA", "BRAV3.SA", "BPAC11.SA",
    "BHIA3.SA", "CMIG4.SA", "COGN3.SA", "CPLE3.SA", "CSAN3.SA", "CPFE3.SA", "CMIN3.SA",
    "CURY3.SA", "CVCB3.SA", "CYRE3.SA", "ECOR3.SA", "ENGI11.SA", "ENEV3.SA", "EGIE3.SA",
    "EQTL3.SA", "FLRY3.SA", "GGBR4.SA", "GOAU4.SA", "GGPS3.SA", "HAPV3.SA", "HYPE3.SA",
    "IGTI11.SA", "IRBR3.SA", "ITSA4.SA", "ITUB4.SA", "KLBN11.SA", "RENT3.SA", "LREN3.SA",
    "MGLU3.SA", "BEEF3.SA", "MOVI3.SA", "MRVE3.SA", "MULT3.SA", "NATU3.SA", "NEOE3.SA",
    "PETR3.SA", "PETR4.SA", "RECV3.SA", "PRIO3.SA", "PSSA3.SA", "RADL3.SA", "RAIZ4.SA",
    "RDOR3.SA", "RAIL3.SA", "SBSP3.SA", "SANB11.SA", "CSNA3.SA", "SLCE3.SA", "SUZB3.SA",
    "TAEE11.SA", "VIVT3.SA", "TIMS3.SA", "TOTS3.SA", "UGPA3.SA", "USIM5.SA", "VALE3.SA",
    "VBBR3.SA", "WEGE3.SA", "YDUQ3.SA"
]

def atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()

# Baixa dados
dados = {}
for t in TICKERS:
    try:
        df = yf.download(t, start=START, progress=False)
        if len(df) > 100:
            df['ATR'] = atr(df)
            df['HH20'] = df['High'].rolling(20).max().shift(1)
            dados[t] = df
            print(f"Dados baixados para {t}: {len(df)} registros")
    except Exception as e:
        print(f"Erro ao baixar dados de {t}: {e}")
        continue

equity = CAPITAL_INICIAL
posicoes = []
trades = []

if dados:
    datas = sorted(set().union(*[df.index for df in dados.values()]))
    print(f"Total de datas: {len(datas)}")

    for data in datas:
        # 1️⃣ Atualiza posições abertas
        novas_posicoes = []
        for pos in posicoes:
            df = dados[pos['ticker']]
            if data not in df.index:
                novas_posicoes.append(pos)
                continue

            try:
                low_val = df.loc[data, 'Low']
                if isinstance(low_val, pd.Series):
                    low_val = low_val.iloc[0]

                # Verifica se o stop foi atingido
                if low_val <= pos['stop']:
                    # CORREÇÃO: Devolve o valor total da venda, não apenas o lucro
                    exit_price = pos['stop']  # Saída no stop
                    valor_venda = pos['shares'] * exit_price
                    
                    # Calcula o P&L para registro
                    valor_compra = pos['shares'] * pos['entry']
                    pnl = valor_venda - valor_compra
                    
                    # Atualiza o equity com o valor total da venda
                    equity += valor_venda
                    
                    # Registra o trade
                    trades.append({
                        'ticker': pos['ticker'],
                        'entry_date': pos.get('entry_date', 'Unknown'),
                        'exit_date': data,
                        'entry_price': pos['entry'],
                        'exit_price': exit_price,
                        'shares': pos['shares'],
                        'pnl': pnl,
                        'pnl_percent': (pnl / valor_compra) * 100 if valor_compra > 0 else 0,
                        'type': 'loss' if pnl < 0 else 'gain'
                    })
                    
                    print(f"Saída: {pos['ticker']} - Data: {data} - Preço: {exit_price:.2f} - P&L: R$ {pnl:,.2f} ({pnl/valor_compra*100:.2f}%)")
                else:
                    novas_posicoes.append(pos)
            except Exception as e:
                print(f"Erro ao processar posição {pos['ticker']} na data {data}: {e}")
                novas_posicoes.append(pos)

        posicoes = novas_posicoes

        # 2️⃣ Novas entradas
        for ticker, df in dados.items():
            if len(posicoes) >= MAX_POSICOES:
                break

            if data not in df.index:
                continue

            try:
                close_val = df.loc[data, 'Close']
                if isinstance(close_val, pd.Series):
                    close_val = close_val.iloc[0]
                
                hh20_val = df.loc[data, 'HH20']
                if isinstance(hh20_val, pd.Series):
                    hh20_val = hh20_val.iloc[0]
                
                atr_val = df.loc[data, 'ATR']
                if isinstance(atr_val, pd.Series):
                    atr_val = atr_val.iloc[0]

                if pd.isna(hh20_val) or pd.isna(atr_val):
                    continue

                if close_val > hh20_val and not any(p['ticker'] == ticker for p in posicoes):
                    risk_amount = equity * RISK_PER_TRADE
                    stop = close_val - 2 * atr_val
                    risk_por_acao = close_val - stop

                    if risk_por_acao <= 0 or pd.isna(risk_por_acao):
                        continue

                    shares = risk_amount / risk_por_acao
                    custo = shares * close_val

                    if custo > equity:
                        shares = equity // close_val
                        custo = shares * close_val

                    if shares > 0:
                        # CORREÇÃO: Subtrai o valor total da compra do equity
                        equity -= custo
                        posicoes.append({
                            'ticker': ticker,
                            'entry': close_val,
                            'stop': stop,
                            'shares': shares,
                            'entry_date': data,
                            'cost': custo  # Armazena o custo total para referência
                        })
                        print(f"Entrada: {ticker} - Data: {data} - Preço: {close_val:.2f} - Custo: R$ {custo:,.2f} - Shares: {shares:.0f}")
            except Exception as e:
                print(f"Erro ao processar entrada para {ticker} na data {data}: {e}")
                continue

    # Processa posições que sobraram no final (vende no último preço disponível)
    for pos in posicoes:
        df = dados[pos['ticker']]
        last_date = df.index[-1]
        last_close = df.loc[last_date, 'Close']
        if isinstance(last_close, pd.Series):
            last_close = last_close.iloc[0]
        
        valor_venda = pos['shares'] * last_close
        valor_compra = pos['shares'] * pos['entry']
        pnl = valor_venda - valor_compra
        
        equity += valor_venda
        
        trades.append({
            'ticker': pos['ticker'],
            'entry_date': pos['entry_date'],
            'exit_date': last_date,
            'entry_price': pos['entry'],
            'exit_price': last_close,
            'shares': pos['shares'],
            'pnl': pnl,
            'pnl_percent': (pnl / valor_compra) * 100 if valor_compra > 0 else 0,
            'type': 'loss' if pnl < 0 else 'gain'
        })
        
        print(f"Saída final: {pos['ticker']} - Data: {last_date} - Preço: {last_close:.2f} - P&L: R$ {pnl:,.2f}")

    print("\n" + "="*50)
    print("📊 RESULTADO DO BACKTEST")
    print("="*50)
    print(f"Capital inicial: R$ {CAPITAL_INICIAL:,.2f}")
    print(f"Capital final:   R$ {equity:,.2f}")
    
    if trades:
        trades_df = pd.DataFrame(trades)
        total_pnl = trades_df['pnl'].sum()
        retorno_percentual = (total_pnl / CAPITAL_INICIAL) * 100
        
        print(f"Total P&L:       R$ {total_pnl:,.2f} ({retorno_percentual:,.2f}%)")
        print(f"Total de trades: {len(trades_df)}")
        
        wins = len(trades_df[trades_df['pnl'] > 0])
        losses = len(trades_df[trades_df['pnl'] < 0])
        
        print(f"Wins:  {wins}")
        print(f"Losses: {losses}")
        print(f"Winrate: {wins/len(trades_df)*100:.2f}%" if len(trades_df) > 0 else "Winrate: 0%")
        
        if wins > 0:
            print(f"Maior gain: R$ {trades_df[trades_df['pnl'] > 0]['pnl'].max():,.2f}")
            print(f"Média gain: R$ {trades_df[trades_df['pnl'] > 0]['pnl'].mean():,.2f}")
        
        if losses > 0:
            print(f"Maior loss: R$ {trades_df[trades_df['pnl'] < 0]['pnl'].min():,.2f}")
            print(f"Média loss: R$ {trades_df[trades_df['pnl'] < 0]['pnl'].mean():,.2f}")
        
        # Fator de lucro (Gross Profit / Gross Loss)
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        if gross_loss > 0:
            profit_factor = gross_profit / gross_loss
            print(f"Profit Factor: {profit_factor:.2f}")
        
        print("="*50)
    else:
        print("Nenhum trade foi realizado.")
else:
    print("Nenhum dado foi baixado com sucesso.")