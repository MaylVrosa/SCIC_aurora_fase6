"""
erros.py
========
Análise numérica do SCIC:

* erro absoluto e erro relativo entre a latência prevista (estimativa de
  engenharia) e a latência observada;
* faixas de aceitação (aceitável, atenção, crítico), mais rígidas para os
  módulos que mantêm pessoas vivas;
* comparação entre escalas (enlaces internos em ms x enlace Terra-Marte em min);
* experimentos de ponto flutuante (IEEE 754);
* simulação da queda de qualidade do sinal pelo método de Euler, comparada
  com a solução exata, e o método de Newton para achar o sol em que o sinal
  cruza o limite de 70%.
"""
from __future__ import annotations

import math
import struct
import sys
from decimal import Decimal
from pathlib import Path

import numpy as np
import pandas as pd

import graficos as g
from dados import eh_interplanetario

# limites do erro relativo (%): (aceitável até, atenção até); acima = crítico
LIMITES_PADRAO = (10.0, 25.0)
LIMITES_VIDA_HUMANA = (5.0, 15.0)      # prioridade 5: médico, suporte de vida, comando...
LIMITE_QUALIDADE_PCT = 70.0
MODULO_SINAL = "Estufa Hidropônica"


# --------------------------------------------------------------------------
# erro absoluto e relativo
# --------------------------------------------------------------------------
def classificar_erro(erro_rel_pct: float, prioridade_base: int) -> str:
    """Classifica o erro relativo usando limites mais rígidos para vida humana."""
    if pd.isna(erro_rel_pct):
        return "sem referência"
    aceitavel, atencao = LIMITES_VIDA_HUMANA if prioridade_base >= 5 else LIMITES_PADRAO
    if erro_rel_pct <= aceitavel:
        return "aceitável"
    if erro_rel_pct <= atencao:
        return "atenção"
    return "crítico"


def calcular_erros(df: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta erro_abs_ms, erro_rel_pct e faixa_erro a cada registro.

    erro absoluto = |prevista - observada|            (em ms, mesma unidade da medida)
    erro relativo = erro absoluto / |observada| x 100  (adimensional, em %)
    A referência é a latência observada (o valor real medido). Observada igual
    a zero vira vazio, para não dividir por zero.
    """
    df = df.copy()
    df["erro_abs_ms"] = (df["latencia_prevista_ms"] - df["latencia_observada_ms"]).abs()
    referencia = df["latencia_observada_ms"].abs().replace(0, np.nan)
    df["erro_rel_pct"] = df["erro_abs_ms"] / referencia * 100
    df["faixa_erro"] = [classificar_erro(e, p) for e, p in
                        zip(df["erro_rel_pct"], df["prioridade_base"])]
    return df


def resumo_erros_por_modulo(df: pd.DataFrame) -> pd.DataFrame:
    """Erro absoluto médio, erro relativo médio e máximo, e % de registros críticos."""
    agrupado = df.groupby("modulo")
    tabela = pd.DataFrame({
        "prioridade": agrupado["prioridade_base"].first(),
        "erro_abs_medio_ms": agrupado["erro_abs_ms"].mean(),
        "erro_rel_medio_pct": agrupado["erro_rel_pct"].mean(),
        "erro_rel_max_pct": agrupado["erro_rel_pct"].max(),
        "criticos_pct": agrupado["faixa_erro"].apply(lambda s: (s == "crítico").mean() * 100),
    })
    tabela["faixa_media"] = [classificar_erro(e, p) for e, p in
                             zip(tabela["erro_rel_medio_pct"], tabela["prioridade"])]
    return tabela.sort_values("erro_rel_medio_pct", ascending=False)


def comparar_escalas(df: pd.DataFrame) -> dict:
    """Mostra por que o erro relativo é necessário: escalas muito diferentes."""
    internos = df[~eh_interplanetario(df)]
    terra = df[eh_interplanetario(df)]
    return {
        "interno_lat_media_ms": float(internos["latencia_observada_ms"].mean()),
        "interno_erro_abs_ms": float(internos["erro_abs_ms"].mean()),
        "interno_erro_rel_pct": float(internos["erro_rel_pct"].mean()),
        "terra_lat_media_ms": float(terra["latencia_observada_ms"].mean()),
        "terra_erro_abs_ms": float(terra["erro_abs_ms"].mean()),
        "terra_erro_rel_pct": float(terra["erro_rel_pct"].mean()),
        "vies_interno_ms": float((internos["latencia_observada_ms"]
                                  - internos["latencia_prevista_ms"]).mean()),
    }


# --------------------------------------------------------------------------
# ponto flutuante
# --------------------------------------------------------------------------
def decompor_ieee754(valor: float) -> dict:
    """Separa um float de 64 bits em sinal (1 bit), expoente (11) e mantissa (52)."""
    bits = "".join(f"{byte:08b}" for byte in struct.pack(">d", valor))
    sinal, expoente, mantissa = bits[0], bits[1:12], bits[12:]
    reconstruido = ((-1) ** int(sinal)) * (1 + int(mantissa, 2) / 2 ** 52) \
        * 2 ** (int(expoente, 2) - 1023)
    return {"valor": valor, "sinal": sinal, "expoente": expoente, "mantissa": mantissa,
            "expoente_real": int(expoente, 2) - 1023, "reconstruido": reconstruido,
            "valor_exato_guardado": str(Decimal(valor))}


def demonstrar_ponto_flutuante() -> dict:
    """Experimentos que mostram os limites da representação em ponto flutuante."""
    soma_dez = sum([0.1] * 10)
    potencia = 28 * 1.6
    exato_01 = Decimal(0.1)
    return {
        "soma_01_02": 0.1 + 0.2,
        "soma_igual_03": (0.1 + 0.2) == 0.3,
        "isclose_03": math.isclose(0.1 + 0.2, 0.3),
        "soma_dez_vezes_01": soma_dez,
        "potencia_28x16": potencia,
        "potencia_arredondada": round(potencia, 2),
        "erro_abs_de_01": float(abs(exato_01 - Decimal("0.1"))),
        "erro_rel_de_01": float(abs(exato_01 - Decimal("0.1")) / Decimal("0.1")),
        "epsilon": sys.float_info.epsilon,
        "grande_mais_1": (1e16 + 1) - 1e16,
        "ieee_latencia": decompor_ieee754(72.2),
    }


def propagacao_arredondamento(df: pd.DataFrame, meia_unidade: float = 0.005) -> dict:
    """Quanto o arredondamento do CSV (2 casas em V e I) pode mudar a potência P = V x I.

    Se V e I podem estar errados em até 0,005, o erro máximo de P é aproximadamente
    |I| x 0,005 + |V| x 0,005 (regra de propagação de erros para um produto).
    Calculado para o módulo de maior potência média, o caso mais desfavorável em W.
    """
    medias = df.groupby("modulo")[["tensao_v", "corrente_a"]].mean()
    potencias = medias["tensao_v"] * medias["corrente_a"]
    modulo = potencias.idxmax()
    tensao, corrente = medias.loc[modulo, "tensao_v"], medias.loc[modulo, "corrente_a"]
    potencia = tensao * corrente
    erro_max = abs(corrente) * meia_unidade + abs(tensao) * meia_unidade
    return {"modulo": modulo, "tensao_v": float(tensao), "corrente_a": float(corrente),
            "potencia_w": float(potencia), "erro_max_w": float(erro_max),
            "erro_rel_pct": float(erro_max / potencia * 100), "meia_unidade": meia_unidade}


# --------------------------------------------------------------------------
# Euler e Newton: queda da qualidade do sinal da Estufa
# --------------------------------------------------------------------------
def estimar_taxa_queda(df: pd.DataFrame, modulo: str = MODULO_SINAL) -> tuple[float, float]:
    """Ajusta Q(t) = Q0 * e^(-taxa * t) aos dados do módulo (reta em ln Q)."""
    serie = df[df["modulo"] == modulo].sort_values("ciclo")
    t = serie["ciclo"].to_numpy(dtype=float) - 1
    q = serie["qualidade_sinal_pct"].to_numpy(dtype=float)
    inclinacao, intercepto = np.polyfit(t, np.log(q), 1)
    return float(-inclinacao), float(math.exp(intercepto))


def simular_euler(q0: float, taxa: float, passo: float, t_final: float) -> tuple[np.ndarray, np.ndarray]:
    """Resolve dQ/dt = -taxa * Q pelo método de Euler: Q(n+1) = Q(n) + h * f(Q(n))."""
    n_passos = int(round(t_final / passo))
    tempos = np.linspace(0, n_passos * passo, n_passos + 1)
    valores = np.empty(n_passos + 1)
    valores[0] = q0
    for n in range(n_passos):
        valores[n + 1] = valores[n] + passo * (-taxa * valores[n])
    return tempos, valores


def newton_cruzamento(q0: float, taxa: float, limite: float = LIMITE_QUALIDADE_PCT,
                      chute: float = 0.0, tolerancia: float = 1e-8) -> tuple[float, int]:
    """Método de Newton para f(t) = Q0 * e^(-taxa t) - limite = 0."""
    t = chute
    for iteracao in range(1, 51):
        f = q0 * math.exp(-taxa * t) - limite
        derivada = -taxa * q0 * math.exp(-taxa * t)
        novo_t = t - f / derivada
        if abs(novo_t - t) < tolerancia:
            return novo_t, iteracao
        t = novo_t
    return t, 50


def analise_euler(df: pd.DataFrame, passos=(2.0, 1.0, 0.5, 0.1), t_final: float = 28.0) -> dict:
    """Compara Euler com a solução exata e acha o sol em que o sinal cruza 70%.

    t_final = 28 sóis (do sol 1 ao sol 29) é múltiplo de todos os passos, então
    cada simulação termina exatamente no mesmo instante da solução exata.
    """
    taxa, q0 = estimar_taxa_queda(df)
    exato_final = q0 * math.exp(-taxa * t_final)
    linhas = []
    for h in passos:
        _, valores = simular_euler(q0, taxa, h, t_final)
        erro_abs = abs(valores[-1] - exato_final)
        linhas.append({"passo_h": h, "euler_final": valores[-1], "exato_final": exato_final,
                       "erro_abs": erro_abs, "erro_rel_pct": erro_abs / exato_final * 100})
    tabela = pd.DataFrame(linhas)
    t_newton, iteracoes = newton_cruzamento(q0, taxa)
    t_analitico = math.log(q0 / LIMITE_QUALIDADE_PCT) / taxa
    return {
        "modulo": MODULO_SINAL,
        "taxa_por_sol": taxa,
        "q0": q0,
        "tabela": tabela,
        "t_cruzamento_newton": t_newton,
        "iteracoes_newton": iteracoes,
        "t_cruzamento_analitico": t_analitico,
        "sol_cruzamento": t_newton + 1,      # t = 0 corresponde ao sol 1
    }


# --------------------------------------------------------------------------
# gráficos
# --------------------------------------------------------------------------
def grafico_erros_relativos(df: pd.DataFrame) -> Path:
    """Erro relativo médio por módulo, colorido pela faixa de aceitação."""
    tabela = resumo_erros_por_modulo(df).sort_values("erro_rel_medio_pct")
    fig, ax = g.nova_figura(8, 4.8)
    y = np.arange(len(tabela))
    cores = [g.STATUS_COR.get(f, g.TINTA_FRACA) for f in tabela["faixa_media"]]
    ax.barh(y, tabela["erro_rel_medio_pct"], height=0.55, color=cores)
    for posicao, valor, faixa in zip(y, tabela["erro_rel_medio_pct"], tabela["faixa_media"]):
        ax.text(valor + 0.4, posicao, f"{g.fmt(valor)}%  ({faixa})", va="center", fontsize=8.5,
                color=g.TINTA_2)
    ax.set_yticks(y)
    ax.set_yticklabels(tabela.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, tabela["erro_rel_medio_pct"].max() * 1.45)
    ax.set_xlabel("Erro relativo médio da estimativa de engenharia (%)")
    criticos = int((tabela["faixa_media"] == "crítico").sum())
    ax.set_title(f"{criticos} módulos têm erro médio na faixa crítica")
    g.subtitulo(ax, "Limites: 10% e 25% (5% e 15% para módulos que mantêm vidas)")
    return g.salvar_figura(fig, "erros_relativos_por_modulo.png")


def grafico_euler(df: pd.DataFrame, resultado: dict) -> Path:
    """Painel 1: medições, solução exata, Euler e Newton. Painel 2: erro de Euler por passo."""
    serie = df[df["modulo"] == MODULO_SINAL].sort_values("ciclo")
    q0, taxa = resultado["q0"], resultado["taxa_por_sol"]
    sol_cruzamento = resultado["sol_cruzamento"]
    fig, (ax1, ax2) = g.nova_figura(11, 4.6, paineis=2)

    ax1.scatter(serie["ciclo"], serie["qualidade_sinal_pct"], s=24, color=g.CINZA_CONTEXTO,
                label="Medições da Estufa", zorder=2)
    t_fino = np.linspace(0, 29, 300)
    ax1.plot(t_fino + 1, q0 * np.exp(-taxa * t_fino), color=g.TINTA, linewidth=2,
             label="Solução exata", zorder=3)
    tempos, valores = simular_euler(q0, taxa, 4.0, 28.0)
    ax1.plot(tempos + 1, valores, "o--", color=g.LARANJA, markersize=5, linewidth=1.4,
             markeredgecolor="white", label="Euler, h = 4 sóis", zorder=4)
    ax1.axhline(LIMITE_QUALIDADE_PCT, color=g.STATUS_COR["crítico"], linewidth=1)
    ax1.text(1, LIMITE_QUALIDADE_PCT + 0.8, "limite de 70%", color=g.TINTA_2, fontsize=8.5)
    ax1.axvline(sol_cruzamento, color=g.TINTA_FRACA, linewidth=1)
    ax1.text(sol_cruzamento + 0.4, 95, f"Newton:\nsol {g.fmt(sol_cruzamento)}", color=g.TINTA_2,
             fontsize=8.5, va="top")
    ax1.set_ylim(55, 100)
    ax1.set_xlabel("Sol")
    ax1.set_ylabel("Qualidade do sinal (%)")
    ax1.set_title(f"O sinal da Estufa cruza 70% no sol {g.fmt(sol_cruzamento)}")
    g.subtitulo(ax1, f"dQ/dt = −λQ com λ = {g.fmt(taxa, 4)} por sol, ajustado às medições")
    ax1.legend(loc="lower left")

    for h, cor in zip((4.0, 2.0, 1.0), (g.LARANJA, g.VERDE_AGUA, g.AZUL)):
        tempos, valores = simular_euler(q0, taxa, h, 28.0)
        erro = np.abs(valores - q0 * np.exp(-taxa * tempos))
        ax2.plot(tempos + 1, erro, "o-", color=cor, markersize=4.5, linewidth=1.6,
                 markeredgecolor="white", label=f"h = {h:g} {'sol' if h == 1 else 'sóis'}")
        ax2.text(tempos[-1] + 1.6, erro[-1], g.fmt(erro[-1], 2), va="center", fontsize=8.5,
                 color=g.TINTA_2)
    ax2.set_xlim(0, 33)
    ax2.set_xlabel("Sol")
    ax2.set_ylabel("|Euler − exata| (pontos de qualidade)")
    ax2.set_title("Passo pela metade, erro pela metade")
    g.subtitulo(ax2, "Erro de truncamento do método de Euler, de 1ª ordem")
    ax2.legend(loc="upper left")
    fig.tight_layout(w_pad=3)
    return g.salvar_figura(fig, "euler_qualidade_sinal.png")
