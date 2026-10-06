"""
gestao_inteligente.py
=====================
Gerenciamento inteligente da comunicação, calculado com os dados do SCIC:

* monitoramento contínuo: cada leitura é comparada com a média móvel dos sóis
  anteriores do próprio módulo; acima de média + 2 desvios-padrão é anomalia;
* evento sistêmico x falha local: sóis em que vários módulos entram em alerta
  ao mesmo tempo (como a tempestade de poeira) pedem resposta diferente de uma
  falha isolada;
* manutenção preditiva: a tendência da qualidade do sinal de cada módulo
  indica quando ele vai cruzar o limite de 70%;
* redundância: quanto um enlace reserva aumenta a disponibilidade;
* eficiência: energia gasta com retransmissões causadas pela perda de pacotes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import graficos as g
from dados import eh_interplanetario

JANELA_MOVEL = 5           # sóis usados na média móvel
FATOR_DESVIO = 2.0         # anomalia = acima de média + 2 desvios
LIMITE_QUALIDADE = 70.0
INCLINACAO_PREOCUPANTE = -0.3   # pontos de qualidade perdidos por sol
MODULOS_ENLACE_PRINCIPAL = ("Comunicação Orbital", "Antena Espaço Profundo")
PERDA_ALVO_PCT = 0.5       # meta de perda de pacotes após manutenção


def detectar_anomalias(df: pd.DataFrame) -> pd.DataFrame:
    """Latência acima de média móvel + 2 desvios (só com sóis anteriores, sem olhar o futuro)."""
    internos = df[~eh_interplanetario(df)].sort_values(["modulo", "ciclo"]).copy()
    grupos = internos.groupby("modulo")["latencia_observada_ms"]
    internos["media_movel"] = grupos.transform(
        lambda s: s.shift(1).rolling(JANELA_MOVEL, min_periods=JANELA_MOVEL).mean())
    internos["desvio_movel"] = grupos.transform(
        lambda s: s.shift(1).rolling(JANELA_MOVEL, min_periods=JANELA_MOVEL).std())
    internos["limite_superior"] = internos["media_movel"] + FATOR_DESVIO * internos["desvio_movel"]
    internos["anomalia"] = internos["latencia_observada_ms"] > internos["limite_superior"]
    return internos


def eventos_sistemicos(df: pd.DataFrame, minimo_modulos: int = 5) -> pd.DataFrame:
    """Sóis em que muitos módulos entram em alerta juntos (causa comum, não falha local)."""
    por_sol = df.groupby("ciclo").agg(
        modulos_em_alerta=("status", lambda s: int((s == "alerta").sum())),
        qualidade_media=("qualidade_sinal_pct", "mean"),
        perda_media=("perda_pacotes_pct", "mean"),
    )
    por_sol["evento_sistemico"] = por_sol["modulos_em_alerta"] >= minimo_modulos
    return por_sol


def tendencia_degradacao(df: pd.DataFrame) -> pd.DataFrame:
    """Inclinação da qualidade do sinal por sol (reta de mínimos quadrados) e previsão."""
    linhas = []
    ultimo_sol = int(df["ciclo"].max())
    for modulo, grupo in df.groupby("modulo"):
        inclinacao, intercepto = np.polyfit(grupo["ciclo"], grupo["qualidade_sinal_pct"], 1)
        atual = intercepto + inclinacao * ultimo_sol
        if inclinacao < 0:
            sol_cruzamento = (LIMITE_QUALIDADE - intercepto) / inclinacao
        else:
            sol_cruzamento = np.nan
        if inclinacao <= INCLINACAO_PREOCUPANTE:
            recomendacao = ("manutenção corretiva já" if atual < LIMITE_QUALIDADE
                            else "agendar manutenção preditiva")
        else:
            recomendacao = "estável"
        linhas.append({"modulo": modulo, "inclinacao_por_sol": inclinacao,
                       "qualidade_tendencia_atual": atual, "sol_cruza_70": sol_cruzamento,
                       "recomendacao": recomendacao})
    return pd.DataFrame(linhas).set_index("modulo").sort_values("inclinacao_por_sol")


def previsao_antecipada(df: pd.DataFrame, modulo: str | None = None, ate_sol: int = 10) -> dict:
    """Refaz a tendência usando só os sóis até `ate_sol`: o cruzamento de 70% já era previsível?

    Mostra o valor da manutenção preditiva: com poucos sóis de dados, a reta já
    aponta quando o sinal vai ficar ruim, com tempo para agendar o reparo.
    """
    if modulo is None:
        modulo = tendencia_degradacao(df).index[0]
    parte = df[(df["modulo"] == modulo) & (df["ciclo"] <= ate_sol)]
    inclinacao, intercepto = np.polyfit(parte["ciclo"], parte["qualidade_sinal_pct"], 1)
    sol_previsto = (LIMITE_QUALIDADE - intercepto) / inclinacao if inclinacao < 0 else np.nan
    return {"modulo": modulo, "ate_sol": ate_sol, "inclinacao_por_sol": float(inclinacao),
            "sol_previsto": float(sol_previsto), "antecedencia_sois": float(sol_previsto - ate_sol)}


def simular_redundancia(df: pd.DataFrame, horas_por_sol: float = 24.66) -> dict:
    """Disponibilidade do enlace com 1 caminho (1 - p) e com 2 independentes (1 - p²).

    p = fração de sóis em que o enlace principal ficou em alerta ou manutenção.
    """
    principal = df[df["modulo"].isin(MODULOS_ENLACE_PRINCIPAL)]
    p = float((principal["status"] != "ativo").mean())
    disp_1, disp_2 = 1 - p, 1 - p ** 2
    sois = int(df["ciclo"].nunique())
    return {
        "p_falha": p,
        "disponibilidade_1_enlace": disp_1,
        "disponibilidade_2_enlaces": disp_2,
        "horas_fora_1_enlace": p * sois * horas_por_sol,
        "horas_fora_2_enlaces": p ** 2 * sois * horas_por_sol,
        "sois_analisados": sois,
    }


def eficiencia_energetica(df: pd.DataFrame) -> dict:
    """Energia gasta com retransmissões: cada pacote perdido é enviado de novo.

    Energia de retransmissão = consumo x perda / (1 - perda). Exige as colunas de
    hardware.calcular_potencia().
    """
    perda = df["perda_pacotes_pct"] / 100
    retransmissao_wh = df["consumo_wh_sol"] * perda / (1 - perda)
    alvo = PERDA_ALVO_PCT / 100
    perda_alvo = perda.clip(upper=alvo)
    retransmissao_alvo_wh = df["consumo_wh_sol"] * perda_alvo / (1 - perda_alvo)
    por_modulo = (retransmissao_wh.groupby(df["modulo"]).sum() / 1000).sort_values(ascending=False)
    return {
        "consumo_total_kwh": float(df["consumo_wh_sol"].sum() / 1000),
        "retransmissao_kwh": float(retransmissao_wh.sum() / 1000),
        "retransmissao_pct": float(retransmissao_wh.sum() / df["consumo_wh_sol"].sum() * 100),
        "economia_possivel_kwh": float((retransmissao_wh.sum() - retransmissao_alvo_wh.sum()) / 1000),
        "por_modulo_kwh": por_modulo,
    }


# --------------------------------------------------------------------------
# gráficos
# --------------------------------------------------------------------------
def grafico_tendencia(df: pd.DataFrame, tendencias: pd.DataFrame) -> Path:
    """Qualidade do sinal: o módulo que mais degrada em destaque, os outros em cinza."""
    pior = tendencias.index[0]
    fig, ax = g.nova_figura(8, 4.6)
    for modulo, grupo in df.groupby("modulo"):
        if modulo != pior:
            ax.plot(grupo["ciclo"], grupo["qualidade_sinal_pct"], color=g.CINZA_CONTEXTO,
                    linewidth=1, alpha=0.8)
    serie = df[df["modulo"] == pior].sort_values("ciclo")
    ax.plot(serie["ciclo"], serie["qualidade_sinal_pct"], "o-", color=g.AZUL, markersize=5,
            markeredgecolor="white", linewidth=2, label=pior)
    linha = tendencias.loc[pior]
    intercepto = linha["qualidade_tendencia_atual"] - linha["inclinacao_por_sol"] * df["ciclo"].max()
    xs = np.array([1, max(linha["sol_cruza_70"], df["ciclo"].max())])
    ax.plot(xs, intercepto + linha["inclinacao_por_sol"] * xs, color=g.LARANJA, linewidth=1.6,
            linestyle="--", label=f"Tendência: {g.fmt(linha['inclinacao_por_sol'], 2)} p.p. por sol")
    ax.axhline(LIMITE_QUALIDADE, color=g.STATUS_COR["crítico"], linewidth=1)
    ax.text(1, LIMITE_QUALIDADE - 2.5, "limite de 70%", fontsize=8.5, color=g.TINTA_2)
    ax.axvspan(11.5, 13.5, color=g.GRADE, alpha=0.6, linewidth=0)
    ax.text(12.5, 99, "tempestade\nde poeira", ha="center", va="top", fontsize=8,
            color=g.TINTA_2)
    ax.set_ylim(55, 100)
    ax.set_xlabel("Sol")
    ax.set_ylabel("Qualidade do sinal (%)")
    ax.set_title(f"{pior}: tendência cruza 70% no sol {linha['sol_cruza_70']:.0f}")
    outros = df[df["modulo"] != pior]
    g.subtitulo(ax, f"Linhas cinza: os outros {outros['modulo'].nunique()} módulos, estáveis em torno "
                    f"de {outros['qualidade_sinal_pct'].mean():.0f}%")
    ax.legend(loc="lower left")
    return g.salvar_figura(fig, "gestao_tendencia_sinal.png")


def grafico_alertas_por_sol(por_sol: pd.DataFrame) -> Path:
    """Módulos em alerta em cada sol; os eventos sistêmicos em destaque."""
    fig, ax = g.nova_figura(8, 4.2)
    cores = [g.LARANJA if evento else g.AZUL for evento in por_sol["evento_sistemico"]]
    ax.bar(por_sol.index, por_sol["modulos_em_alerta"], width=0.6, color=cores)
    for sol, linha in por_sol[por_sol["evento_sistemico"]].iterrows():
        ax.text(sol, linha["modulos_em_alerta"] + 0.15, f"{int(linha['modulos_em_alerta'])}",
                ha="center", fontsize=8.5, color=g.TINTA_2)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Sol")
    ax.set_ylabel("Módulos em alerta")
    eventos = por_sol[por_sol["evento_sistemico"]]
    if len(eventos):
        sois = " e ".join(str(s) for s in eventos.index)
        modulos = " e ".join(str(int(n)) for n in eventos["modulos_em_alerta"])
        ax.set_title(f"Sóis {sois}: {modulos} módulos em alerta ao mesmo tempo")
    else:
        ax.set_title("Nenhum sol com alerta em metade da colônia")
    g.subtitulo(ax, "Laranja = evento sistêmico (5 ou mais módulos, metade da colônia); azul = falhas locais")
    ax.set_xticks(range(1, int(por_sol.index.max()) + 1, 2))
    return g.salvar_figura(fig, "gestao_alertas_por_sol.png")


def grafico_anomalias(anomalias: pd.DataFrame, modulo: str | None = None) -> Path:
    """Latência de um módulo, a média móvel, o limite (média + 2 desvios) e as anomalias."""
    if modulo is None:
        modulo = anomalias.groupby("modulo")["anomalia"].sum().idxmax()
    serie = anomalias[anomalias["modulo"] == modulo]
    fig, ax = g.nova_figura(8, 4.4)
    ax.plot(serie["ciclo"], serie["latencia_observada_ms"], "o-", color=g.AZUL, markersize=5,
            markeredgecolor="white", linewidth=1.8, label="Latência observada")
    ax.plot(serie["ciclo"], serie["media_movel"], color=g.CINZA_CONTEXTO, linewidth=1.6,
            label=f"Média dos {JANELA_MOVEL} sóis anteriores")
    ax.plot(serie["ciclo"], serie["limite_superior"], color=g.LARANJA, linewidth=1.4,
            linestyle="--", label="Limite: média + 2 desvios-padrão")
    pontos = serie[serie["anomalia"]]
    ax.plot(pontos["ciclo"], pontos["latencia_observada_ms"], "o", color=g.STATUS_COR["crítico"],
            markersize=9, markeredgecolor="white", markeredgewidth=1.5,
            label="Anomalia detectada")
    baixo = min(serie["latencia_observada_ms"].min(), serie["media_movel"].min())
    alto = max(serie["latencia_observada_ms"].max(), serie["limite_superior"].max())
    ax.set_ylim(baixo - 5, alto + (alto - baixo) * 0.32)       # espaço para a legenda
    ax.set_xlabel("Sol")
    ax.set_ylabel("Latência (ms)")
    ax.set_title(f"{modulo}: {len(pontos)} leituras acima do limite")
    g.subtitulo(ax, f"O limite usa só sóis anteriores; os {JANELA_MOVEL} primeiros sóis formam a base")
    ax.legend(loc="upper left", ncol=2)
    return g.salvar_figura(fig, "gestao_anomalias.png")
