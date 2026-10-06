"""
hardware.py
===========
Ligação do SCIC com organização e arquitetura de computadores (COA) e com
eletricidade básica aplicada à comunicação:

* conversões manuais entre as bases 2, 10 e 16 (como nas aulas), conferidas
  com as funções prontas do Python;
* decodificação dos códigos de sensor de 16 bits com operações de bits;
* potência dos transmissores (P = V x I), consumo por sol e quedas de tensão;
* Lei de Ohm aplicada a dois componentes do painel: o resistor do LED de
  alerta e o resistor shunt que mede a corrente do transmissor;
* a lista dos dispositivos de entrada, saída e interfaces do sistema.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import graficos as g

TENSAO_NOMINAL_V = 28.0
LIMITE_QUEDA_TENSAO_V = 0.9 * TENSAO_NOMINAL_V   # 25,2 V
HORAS_POR_SOL = 24.66                             # 24 h 39 min 35 s

TIPOS_POR_CODIGO = {
    0x1: "comunicação",
    0x2: "habitação",
    0x3: "agricultura",
    0x4: "laboratório",
    0x5: "suporte médico",
    0x6: "armazenamento de dados",
    0x7: "suporte de vida",
    0x8: "comunicação interplanetária",
}

DIGITOS_HEX = "0123456789ABCDEF"

DISPOSITIVOS_ES = [
    ("Entrada", "Sensor de tensão e corrente (resistor shunt)",
     "mede V e I do transceptor de cada módulo", "barramento I2C até o gateway"),
    ("Entrada", "Medidor de sinal (RSSI) e de perda de pacotes",
     "qualidade do enlace e retransmissões", "rede interna / Wi-Fi"),
    ("Entrada", "Relógio da rede (carimbo de tempo)",
     "latência de ida e volta das mensagens", "rede cabeada"),
    ("Entrada", "Teclado do operador",
     "cadastro de registros e escolha das opções do menu", "USB"),
    ("Saída", "Monitor / terminal do centro de controle",
     "menu, tabelas e análise final", "HDMI"),
    ("Saída", "Gráficos e relatório em PDF", "apoio à decisão e registro histórico",
     "arquivo / impressora"),
    ("Saída", "LED e alarme sonoro no painel",
     "aviso físico do alerta mais urgente (topo do heap)", "GPIO do gateway"),
    ("Interface", "Wi-Fi e Bluetooth (sensores IoT)",
     "levar as leituras dos módulos até o gateway", "sem fio"),
    ("Interface", "Rede cabeada (Ethernet)", "espinha dorsal entre módulos e o centro de controle",
     "cabo"),
    ("Interface", "Rádio de espaço profundo", "enlace com orbitadores e com a Terra",
     "antena de alto ganho"),
]


# --------------------------------------------------------------------------
# bases numéricas (algoritmos manuais das aulas)
# --------------------------------------------------------------------------
def decimal_para_binario(numero: int, bits: int | None = None) -> str:
    """Divisões sucessivas por 2; os restos, lidos de baixo para cima, formam o binário."""
    if numero < 0:
        raise ValueError("use apenas inteiros não negativos")
    if numero == 0:
        resultado = "0"
    else:
        restos = []
        while numero > 0:
            restos.append(str(numero % 2))
            numero //= 2
        resultado = "".join(reversed(restos))
    return resultado.zfill(bits) if bits else resultado


def binario_para_decimal(binario: str) -> int:
    """Soma de potências de 2: cada bit 1 vale 2 elevado à sua posição."""
    binario = binario.replace(" ", "")
    total = 0
    for posicao, bit in enumerate(reversed(binario)):
        if bit not in "01":
            raise ValueError(f"'{bit}' não é um dígito binário")
        total += int(bit) * 2 ** posicao
    return total


def potencias_de_dois(numero: int) -> list[int]:
    """Expoentes dos bits iguais a 1, do mais alto ao mais baixo: 4609 -> [12, 9, 0]."""
    return [i for i in range(numero.bit_length() - 1, -1, -1) if (numero >> i) & 1]


def decimal_para_hexadecimal(numero: int) -> str:
    """Divisões sucessivas por 16, usando a tabela 0-9 e A-F."""
    if numero == 0:
        return "0"
    digitos = []
    while numero > 0:
        digitos.append(DIGITOS_HEX[numero % 16])
        numero //= 16
    return "".join(reversed(digitos))


def hexadecimal_para_binario(hexa: str) -> str:
    """Cada dígito hexadecimal vira exatamente 4 bits."""
    hexa = hexa.upper().replace("0X", "")
    grupos = []
    for digito in hexa:
        if digito not in DIGITOS_HEX:
            raise ValueError(f"'{digito}' não é um dígito hexadecimal")
        grupos.append(decimal_para_binario(DIGITOS_HEX.index(digito), bits=4))
    return " ".join(grupos)


def decodificar_codigo(codigo_hex: str) -> dict:
    """Separa o código de 16 bits em tipo (4 bits), módulo (4 bits) e sensor (8 bits).

    Exemplo: 0x1201 = 0001 0010 0000 0001 -> tipo 1 (comunicação), módulo 2, sensor 1.
    """
    texto = codigo_hex.strip().upper().replace("0X", "")
    if not texto or any(c not in DIGITOS_HEX for c in texto) or len(texto) > 4:
        raise ValueError("informe um código hexadecimal de até 4 dígitos, por exemplo 0x1201")
    binario = hexadecimal_para_binario(texto.zfill(4))
    valor = binario_para_decimal(binario)
    tipo = (valor >> 12) & 0xF          # 4 bits mais altos
    modulo = (valor >> 8) & 0xF         # 4 bits seguintes
    sensor = valor & 0xFF               # 8 bits mais baixos
    return {
        "codigo": f"0x{texto.zfill(4)}",
        "binario": binario,
        "decimal": valor,
        "tipo": tipo,
        "tipo_nome": TIPOS_POR_CODIGO.get(tipo, "tipo desconhecido"),
        "modulo": modulo,
        "sensor": sensor,
        "confere_python": (valor == int(texto, 16)
                           and binario.replace(" ", "") == format(valor, "016b")
                           and decimal_para_hexadecimal(valor).zfill(4) == texto.zfill(4)),
    }


# --------------------------------------------------------------------------
# eletricidade básica
# --------------------------------------------------------------------------
def calcular_potencia(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta potência (P = V x I), consumo por sol e marcação de queda de tensão."""
    df = df.copy()
    df["potencia_w"] = df["tensao_v"] * df["corrente_a"]
    df["consumo_wh_sol"] = df["potencia_w"] * HORAS_POR_SOL
    df["queda_tensao"] = df["tensao_v"] < LIMITE_QUEDA_TENSAO_V
    return df


def resumo_eletrico(df: pd.DataFrame) -> pd.DataFrame:
    """Tensão, corrente, potência e consumo médios por módulo."""
    agrupado = df.groupby("modulo")
    tabela = pd.DataFrame({
        "tensao_media_v": agrupado["tensao_v"].mean(),
        "corrente_media_a": agrupado["corrente_a"].mean(),
        "potencia_media_w": agrupado["potencia_w"].mean(),
        "consumo_30_sois_kwh": agrupado["consumo_wh_sol"].sum() / 1000,
        "quedas_tensao": agrupado["queda_tensao"].sum().astype(int),
    })
    return tabela.sort_values("potencia_media_w", ascending=False)


def lei_de_ohm(tensao: float | None = None, corrente: float | None = None,
               resistencia: float | None = None) -> dict:
    """Resolve V = R x I: informe duas grandezas e a função calcula a terceira."""
    informados = [v is not None for v in (tensao, corrente, resistencia)]
    if sum(informados) != 2:
        raise ValueError("informe exatamente duas grandezas")
    if tensao is None:
        tensao = resistencia * corrente
    elif corrente is None:
        corrente = tensao / resistencia
    else:
        resistencia = tensao / corrente
    return {"tensao_v": tensao, "corrente_a": corrente, "resistencia_ohm": resistencia,
            "potencia_w": tensao * corrente}


def resistor_led(tensao_fonte: float = 5.0, tensao_led: float = 2.0,
                 corrente_led: float = 0.020) -> dict:
    """Resistor que limita a corrente do LED de alerta: R = (Vfonte - Vled) / I."""
    queda = tensao_fonte - tensao_led
    resultado = lei_de_ohm(tensao=queda, corrente=corrente_led)
    resultado["tensao_fonte_v"] = tensao_fonte
    resultado["tensao_led_v"] = tensao_led
    return resultado


def sensor_shunt(corrente: float = 1.6, resistencia_shunt: float = 0.1) -> dict:
    """Resistor shunt em série com o transmissor: a queda V = R x I revela a corrente."""
    return lei_de_ohm(corrente=corrente, resistencia=resistencia_shunt)


def associacao_resistores(resistores: list[float], modo: str = "serie") -> float:
    """Resistência equivalente em série (soma) ou em paralelo (soma dos inversos)."""
    if modo == "serie":
        return float(sum(resistores))
    return float(1 / sum(1 / r for r in resistores))


# --------------------------------------------------------------------------
# gráfico
# --------------------------------------------------------------------------
def grafico_consumo(df: pd.DataFrame) -> Path:
    """Barras horizontais com o consumo de energia de cada módulo em 30 sóis."""
    tabela = resumo_eletrico(df).sort_values("consumo_30_sois_kwh")
    fig, ax = g.nova_figura(8, 4.6)
    y = np.arange(len(tabela))
    cores = [g.AZUL if nome != tabela.index[-1] else g.LARANJA for nome in tabela.index]
    ax.barh(y, tabela["consumo_30_sois_kwh"], height=0.55, color=cores)
    for posicao, valor in zip(y, tabela["consumo_30_sois_kwh"]):
        ax.text(valor + 0.6, posicao, f"{g.fmt(valor)} kWh", va="center", fontsize=8.5,
                color=g.TINTA_2)
    ax.set_yticks(y)
    ax.set_yticklabels(tabela.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Energia consumida pelo transceptor em 30 sóis (kWh)")
    maior = tabela.index[-1]
    fatia = tabela["consumo_30_sois_kwh"].iloc[-1] / tabela["consumo_30_sois_kwh"].sum() * 100
    ax.set_title(f"{maior} consome {fatia:.0f}% da energia de comunicação")
    g.subtitulo(ax, "Consumo = P x 24,66 h por sol, com P = V x I medido em cada registro")
    ax.set_xlim(0, tabela["consumo_30_sois_kwh"].max() * 1.18)
    return g.salvar_figura(fig, "hardware_consumo_por_modulo.png")
