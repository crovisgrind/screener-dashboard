import json
from api.screener import processar_screener

resultado = processar_screener()

with open('resultado.json', 'w', encoding='utf-8') as f:
    json.dump(resultado, f, ensure_ascii=False, indent=2)

print('Salvo em resultado.json!')