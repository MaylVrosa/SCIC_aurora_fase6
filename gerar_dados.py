"""
gerar_dados.py
==============
Gera a base simulada da colônia Aurora Siger: dados_aurora_siger.csv.

A base é sintética, mas segue regras físicas e operacionais simples, para que
as análises do SCIC tenham padrões reais para encontrar:

* a latência dos enlaces internos cresce com a carga da rede e com a perda de
  pacotes, e cai quando a qualidade do sinal melhora;
* o enlace Terra-Marte tem latência dominada pela distância: a luz leva de
  3 a 22 minutos para cruzar o espaço entre os dois planetas;
* a "latência prevista" é a estimativa de engenharia da colônia, que usa a
  carga planejada (55%) e ignora a carga real. Ela é o baseline que o modelo
  de regressão precisa superar;
* alguns problemas foram plantados de propósito: um módulo com o sinal
  degradando, uma tempestade de poeira, quedas de tensão, picos de latência,
  células vazias e uma linha duplicada (para a etapa de limpeza).

Uso:  python gerar_dados.py
A seed é fixa (42), então o arquivo gerado é sempre o mesmo.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_SOIS = 30                       # 1 sol = 1 dia marciano (24 h 39 min 35 s)
VELOCIDADE_LUZ_KM_S = 299_792.458
TENSAO_NOMINAL_V = 28.0           # barramento de energia dos módulos
CARGA_PLANEJADA_PCT = 55.0        # carga usada pela estimativa de engenharia
SOIS_TEMPESTADE = (12, 13)        # tempestade de poeira atinge toda a colônia
MODULO_DEGRADANDO = "Estufa Hidropônica"

ARQUIVO_SAIDA = Path(__file__).resolve().parent / "dados_aurora_siger.csv"

# nome, tipo, código do tipo (4 bits), nº do módulo dentro do tipo,
# prioridade base (5 = vida humana), latência base (ms), distância ao hub (km),
# corrente média do transceptor (A)
MODULOS = [
    ("Comunicação Orbital", "comunicação", 0x1, 1, 4, 30.0, 2.4, 2.6),
    ("Comando Central", "comunicação", 0x1, 2, 5, 18.0, 0.3, 1.8),
    ("Habitação Alfa", "habitação", 0x2, 1, 3, 40.0, 1.2, 1.2),
    ("Habitação Beta", "habitação", 0x2, 2, 3, 45.0, 1.6, 1.2),
    ("Estufa Hidropônica", "agricultura", 0x3, 1, 3, 55.0, 2.1, 1.4),
    ("Laboratório Geológico", "laboratório", 0x4, 1, 2, 70.0, 3.5, 1.5),
    ("Centro Médico", "suporte médico", 0x5, 1, 5, 25.0, 0.8, 1.6),
    ("Data Center", "armazenamento de dados", 0x6, 1, 2, 12.0, 0.5, 2.2),
    ("Controle Ambiental", "suporte de vida", 0x7, 1, 5, 22.0, 0.9, 1.7),
    ("Antena Espaço Profundo", "comunicação interplanetária", 0x8, 1, 5, None, None, 6.5),
]

# (módulo, sol) com queda de tensão no transmissor
QUEDAS_TENSAO = {("Comunicação Orbital", 17): 24.6, ("Centro Médico", 22): 24.9, ("Data Center", 9): 25.0}


def codigo_sensor(cod_tipo: int, n_modulo: int, n_sensor: int = 1) -> str:
    """Monta o código de 16 bits: 4 bits de tipo, 4 de módulo e 8 de sensor."""
    valor = (cod_tipo << 12) | (n_modulo << 8) | n_sensor
    return f"0x{valor:04X}"


def gerar_base(seed: int = SEED) -> pd.DataFrame:
    """Cria o DataFrame com 10 módulos x 30 sóis, mais os problemas plantados."""
    rng = np.random.default_rng(seed)
    linhas = []

    # cada módulo tem um sol de manutenção programada
    sol_manutencao = {m[0]: int(rng.integers(5, 29)) for m in MODULOS}

    # distância Terra-Marte no período: de 225 a 240 milhões de km
    distancias_terra = np.linspace(225.0e6, 240.0e6, N_SOIS)

    for sol in range(1, N_SOIS + 1):
        tempestade = sol in SOIS_TEMPESTADE
        for (nome, tipo, cod_tipo, n_mod, prioridade, base_ms, dist_km, corrente_media) in MODULOS:
            carga = float(np.clip(rng.normal(55, 18), 5, 98))

            # qualidade do sinal: normal ~88%, a Estufa perde sinal ao longo dos sóis
            if nome == MODULO_DEGRADANDO:
                qualidade = 93.0 - 1.0 * (sol - 1) + rng.normal(0, 1.5)
            else:
                qualidade = rng.normal(88, 4)
            if tempestade:
                qualidade -= 12.0
            qualidade = float(np.clip(qualidade, 40, 100))

            perda = 0.12 * (100 - qualidade) - 0.6 + rng.normal(0, 0.3)
            if tempestade:
                perda += 2.0
            perda = float(np.clip(perda, 0.0, 30.0))

            tensao = QUEDAS_TENSAO.get((nome, sol), rng.normal(TENSAO_NOMINAL_V, 0.3))
            corrente = max(0.1, rng.normal(corrente_media, corrente_media * 0.06))

            pico = False
            if tipo == "comunicação interplanetária":
                distancia = float(distancias_terra[sol - 1])
                luz_ms = distancia / VELOCIDADE_LUZ_KM_S * 1000.0
                processamento = rng.normal(1500, 250)
                if sol in (8, 21):               # fila de mensagens acumulada
                    processamento += 3000
                    pico = True
                observada = luz_ms + processamento
                prevista = luz_ms + 1200.0
            else:
                distancia = dist_km
                observada = (base_ms + 0.6 * carga - 0.35 * (qualidade - 85)
                             + 2.0 * perda + rng.normal(0, 4))
                if rng.random() < 0.04:          # rajada de retransmissões
                    observada += rng.uniform(40, 90)
                    pico = True
                prevista = base_ms + 0.5 * CARGA_PLANEJADA_PCT

            # status e mensagem, na ordem de gravidade
            if sol == sol_manutencao[nome]:
                status, mensagem = "manutenção", "Manutenção programada"
            elif tensao < 0.9 * TENSAO_NOMINAL_V:
                status, mensagem = "alerta", "Queda de tensão no transmissor"
            elif qualidade < 70:
                status, mensagem = "alerta", "Qualidade de sinal baixa"
            elif perda > 4.0:
                status, mensagem = "alerta", "Perda de pacotes elevada"
            elif tipo == "comunicação interplanetária" and (observada - prevista) > 2000:
                status, mensagem = "alerta", "Atraso extra no enlace Terra-Marte"
            elif pico or observada > 1.35 * prevista:
                status, mensagem = "alerta", "Latência acima do previsto"
            else:
                status, mensagem = "ativo", ""

            linhas.append({
                "ciclo": sol,
                "modulo": nome,
                "tipo_modulo": tipo,
                "codigo_sensor": codigo_sensor(cod_tipo, n_mod),
                "distancia_km": round(distancia, 1) if distancia < 1e6 else round(distancia),
                "carga_rede_pct": round(carga, 1),
                "qualidade_sinal_pct": round(qualidade, 1),
                "perda_pacotes_pct": round(perda, 2),
                "tensao_v": round(float(tensao), 2),
                "corrente_a": round(float(corrente), 2),
                "latencia_prevista_ms": round(float(prevista), 2),
                "latencia_observada_ms": round(float(observada), 2),
                "status": status,
                "prioridade_base": prioridade,
                "mensagem_alerta": mensagem,
            })

    df = pd.DataFrame(linhas)
    df.insert(0, "id_registro", np.arange(1, len(df) + 1))

    # problemas plantados para a etapa de limpeza
    internos = df.index[df["tipo_modulo"] != "comunicação interplanetária"].to_numpy()
    escolhidos = rng.choice(internos, size=4, replace=False)
    df["qualidade_sinal_pct"] = df["qualidade_sinal_pct"].astype(float)
    df["corrente_a"] = df["corrente_a"].astype(float)
    df["latencia_observada_ms"] = df["latencia_observada_ms"].astype(float)
    df.loc[escolhidos[0], "qualidade_sinal_pct"] = np.nan
    df.loc[escolhidos[1], "qualidade_sinal_pct"] = np.nan
    df.loc[escolhidos[2], "corrente_a"] = np.nan
    df.loc[escolhidos[3], "latencia_observada_ms"] = np.nan
    duplicada = df.iloc[[int(rng.choice(internos))]]
    df = pd.concat([df, duplicada], ignore_index=True)
    return df


def main() -> None:
    df = gerar_base()
    df.to_csv(ARQUIVO_SAIDA, index=False, encoding="utf-8")
    print(f"Base gerada: {ARQUIVO_SAIDA.name} ({len(df)} linhas, {df.shape[1]} colunas)")


if __name__ == "__main__":
    main()
