"""
dados.py
========
Camada de dados do SCIC: leitura e limpeza do CSV, consulta e cadastro de
registros, indicadores de comunicação e gráficos exploratórios.

Todas as funções recebem e devolvem dados (DataFrames, dicionários). Quem
imprime na tela é o menu, em codigo_fonte.py.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

import graficos as g

PASTA_PROJETO = Path(__file__).resolve().parent
ARQUIVO_DADOS = PASTA_PROJETO / "dados_aurora_siger.csv"

TIPO_INTERPLANETARIO = "comunicação interplanetária"
STATUS_VALIDOS = ("ativo", "manutenção", "alerta")

COLUNAS_OBRIGATORIAS = [
    "id_registro", "ciclo", "modulo", "tipo_modulo", "codigo_sensor", "distancia_km",
    "carga_rede_pct", "qualidade_sinal_pct", "perda_pacotes_pct", "tensao_v", "corrente_a",
    "latencia_prevista_ms", "latencia_observada_ms", "status", "prioridade_base",
    "mensagem_alerta",
]
COLUNAS_NUMERICAS = [
    "id_registro", "ciclo", "distancia_km", "carga_rede_pct", "qualidade_sinal_pct",
    "perda_pacotes_pct", "tensao_v", "corrente_a", "latencia_prevista_ms",
    "latencia_observada_ms", "prioridade_base",
]
# colunas que podem ser completadas com a mediana do próprio módulo
COLUNAS_IMPUTAVEIS = ["carga_rede_pct", "qualidade_sinal_pct", "perda_pacotes_pct",
                      "tensao_v", "corrente_a"]
COLUNAS_PERCENTUAIS = ["carga_rede_pct", "qualidade_sinal_pct", "perda_pacotes_pct"]


# --------------------------------------------------------------------------
# utilidades
# --------------------------------------------------------------------------
def normalizar_texto(texto: str) -> str:
    """Minúsculas e sem acentos: 'Comunicação' -> 'comunicacao'."""
    texto = unicodedata.normalize("NFD", str(texto).strip().lower())
    return "".join(c for c in texto if unicodedata.category(c) != "Mn")


def eh_interplanetario(df: pd.DataFrame) -> pd.Series:
    """Máscara das linhas do enlace Terra-Marte (escala de minutos)."""
    return df["tipo_modulo"] == TIPO_INTERPLANETARIO


# --------------------------------------------------------------------------
# leitura e limpeza
# --------------------------------------------------------------------------
def carregar_dados(caminho: Path | str = ARQUIVO_DADOS) -> tuple[pd.DataFrame, dict]:
    """Lê o CSV, limpa os dados e devolve (DataFrame limpo, relatório da limpeza).

    Passos da limpeza:
    1. confere se todas as colunas obrigatórias existem;
    2. converte as colunas numéricas (texto inválido vira vazio);
    3. remove linhas duplicadas;
    4. remove registros sem latência observada (sem ela não há o que analisar);
    5. completa vazios das medições com a mediana do próprio módulo.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Arquivo {caminho.name} não encontrado. Rode 'python gerar_dados.py' para criá-lo.")

    df = pd.read_csv(caminho, encoding="utf-8")
    faltando = [c for c in COLUNAS_OBRIGATORIAS if c not in df.columns]
    if faltando:
        raise ValueError(f"Colunas ausentes no CSV: {', '.join(faltando)}")

    relatorio = {"linhas_lidas": int(len(df))}

    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
    for coluna in ["modulo", "tipo_modulo", "codigo_sensor", "status", "mensagem_alerta"]:
        df[coluna] = df[coluna].fillna("").astype(str).str.strip()
    df["status"] = df["status"].str.lower()
    df["codigo_sensor"] = df["codigo_sensor"].str.upper().str.replace("^0X", "0x", regex=True)

    antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    relatorio["duplicadas_removidas"] = antes - len(df)

    antes = len(df)
    df = df.dropna(subset=["latencia_observada_ms"]).reset_index(drop=True)
    relatorio["sem_latencia_removidas"] = antes - len(df)

    imputados = {}
    for coluna in COLUNAS_IMPUTAVEIS:
        vazios = int(df[coluna].isna().sum())
        if vazios:
            mediana_modulo = df.groupby("modulo")[coluna].transform("median")
            df[coluna] = df[coluna].fillna(mediana_modulo)
            imputados[coluna] = vazios
    relatorio["valores_imputados"] = imputados

    fora_da_faixa = 0
    for coluna in COLUNAS_PERCENTUAIS:
        fora = (df[coluna] < 0) | (df[coluna] > 100)
        fora_da_faixa += int(fora.sum())
        df[coluna] = df[coluna].clip(0, 100)
    relatorio["percentuais_corrigidos"] = fora_da_faixa
    relatorio["status_invalidos"] = int((~df["status"].isin(STATUS_VALIDOS)).sum())

    for coluna in ["id_registro", "ciclo", "prioridade_base"]:
        df[coluna] = df[coluna].astype(int)
    df = df.sort_values(["ciclo", "id_registro"]).reset_index(drop=True)
    relatorio["linhas_finais"] = int(len(df))
    return df, relatorio


def salvar_dados(df: pd.DataFrame, caminho: Path | str = ARQUIVO_DADOS) -> Path:
    """Grava a base no CSV (UTF-8, separador vírgula, decimal com ponto)."""
    colunas = [c for c in COLUNAS_OBRIGATORIAS if c in df.columns]
    df[colunas].to_csv(caminho, index=False, encoding="utf-8")
    return Path(caminho)


# --------------------------------------------------------------------------
# consulta e cadastro
# --------------------------------------------------------------------------
def consultar_registros(df: pd.DataFrame, modulo: str | None = None, status: str | None = None,
                        ciclo: int | None = None) -> pd.DataFrame:
    """Filtra registros por módulo (busca parcial, sem acento), status e/ou ciclo."""
    resultado = df
    if modulo:
        alvo = normalizar_texto(modulo)
        nomes = resultado["modulo"].map(normalizar_texto)
        resultado = resultado[nomes.str.contains(alvo, regex=False)]
    if status:
        resultado = resultado[resultado["status"] == normalizar_texto(status).replace(
            "manutencao", "manutenção")]
    if ciclo is not None:
        resultado = resultado[resultado["ciclo"] == int(ciclo)]
    return resultado


def validar_registro(registro: dict) -> list[str]:
    """Confere os campos de um novo registro e devolve a lista de problemas."""
    problemas = []
    if int(registro.get("ciclo", 0)) < 1:
        problemas.append("ciclo deve ser um inteiro maior ou igual a 1")
    for campo in COLUNAS_PERCENTUAIS:
        valor = float(registro.get(campo, -1))
        if not 0 <= valor <= 100:
            problemas.append(f"{campo} deve estar entre 0 e 100")
    if float(registro.get("tensao_v", 0)) <= 0:
        problemas.append("tensao_v deve ser positiva")
    if float(registro.get("corrente_a", -1)) < 0:
        problemas.append("corrente_a não pode ser negativa")
    for campo in ["latencia_prevista_ms", "latencia_observada_ms"]:
        if float(registro.get(campo, 0)) <= 0:
            problemas.append(f"{campo} deve ser positiva")
    if registro.get("status") not in STATUS_VALIDOS:
        problemas.append("status deve ser ativo, manutenção ou alerta")
    return problemas


def cadastrar_registro(df: pd.DataFrame, registro: dict) -> pd.DataFrame:
    """Acrescenta um registro de um módulo já existente e devolve a nova base.

    Tipo, código do sensor, distância e prioridade são herdados do módulo,
    para evitar inconsistências de digitação.
    """
    alvo = normalizar_texto(registro.get("modulo", ""))
    candidatos = df[df["modulo"].map(normalizar_texto) == alvo]
    if candidatos.empty:
        raise ValueError(f"Módulo '{registro.get('modulo')}' não existe na base.")
    problemas = validar_registro(registro)
    if problemas:
        raise ValueError("; ".join(problemas))

    referencia = candidatos.iloc[-1]
    novo = {
        "id_registro": int(df["id_registro"].max()) + 1,
        "ciclo": int(registro["ciclo"]),
        "modulo": referencia["modulo"],
        "tipo_modulo": referencia["tipo_modulo"],
        "codigo_sensor": referencia["codigo_sensor"],
        "distancia_km": referencia["distancia_km"],
        "carga_rede_pct": float(registro["carga_rede_pct"]),
        "qualidade_sinal_pct": float(registro["qualidade_sinal_pct"]),
        "perda_pacotes_pct": float(registro["perda_pacotes_pct"]),
        "tensao_v": float(registro["tensao_v"]),
        "corrente_a": float(registro["corrente_a"]),
        "latencia_prevista_ms": float(registro["latencia_prevista_ms"]),
        "latencia_observada_ms": float(registro["latencia_observada_ms"]),
        "status": registro["status"],
        "prioridade_base": int(referencia["prioridade_base"]),
        "mensagem_alerta": str(registro.get("mensagem_alerta", "")).strip(),
    }
    return pd.concat([df, pd.DataFrame([novo])], ignore_index=True)


# --------------------------------------------------------------------------
# indicadores
# --------------------------------------------------------------------------
def calcular_indicadores(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Indicadores de comunicação por módulo e da colônia inteira.

    Disponibilidade = % de sóis em que o módulo esteve 'ativo' (sem alerta e
    sem manutenção). Potência e consumo exigem as colunas criadas por
    hardware.calcular_potencia().
    """
    agrupado = df.groupby("modulo", sort=False)
    tabela = pd.DataFrame({
        "tipo": agrupado["tipo_modulo"].first(),
        "latencia_media_ms": agrupado["latencia_observada_ms"].mean(),
        "qualidade_media_pct": agrupado["qualidade_sinal_pct"].mean(),
        "perda_media_pct": agrupado["perda_pacotes_pct"].mean(),
        "disponibilidade_pct": agrupado["status"].apply(lambda s: (s == "ativo").mean() * 100),
        "alertas": agrupado["status"].apply(lambda s: int((s == "alerta").sum())),
    })
    if "potencia_w" in df.columns:
        tabela["potencia_media_w"] = agrupado["potencia_w"].mean()
        tabela["consumo_30_sois_kwh"] = agrupado["consumo_wh_sol"].sum() / 1000
    tabela = tabela.sort_values("disponibilidade_pct", kind="mergesort")   # ordenação estável

    internos = df[~eh_interplanetario(df)]
    terra = df[eh_interplanetario(df)]
    resumo = {
        "registros": int(len(df)),
        "modulos": int(df["modulo"].nunique()),
        "sois": int(df["ciclo"].nunique()),
        "status": df["status"].value_counts().to_dict(),
        "pct_alerta": float((df["status"] == "alerta").mean() * 100),
        "latencia_media_interna_ms": float(internos["latencia_observada_ms"].mean()),
        "latencia_media_terra_min": float(terra["latencia_observada_ms"].mean() / 60000)
        if len(terra) else float("nan"),
        "latencia_min_terra_min": float(terra["latencia_observada_ms"].min() / 60000)
        if len(terra) else float("nan"),
        "latencia_max_terra_min": float(terra["latencia_observada_ms"].max() / 60000)
        if len(terra) else float("nan"),
        "disponibilidade_media_pct": float(tabela["disponibilidade_pct"].mean()),
        "pior_disponibilidade": (str(tabela.index[0]), float(tabela["disponibilidade_pct"].iloc[0])),
    }
    if "consumo_wh_sol" in df.columns:
        resumo["consumo_total_kwh"] = float(df["consumo_wh_sol"].sum() / 1000)
    return tabela, resumo


# --------------------------------------------------------------------------
# gráficos exploratórios
# --------------------------------------------------------------------------
def gerar_graficos_exploratorios(df: pd.DataFrame) -> list[Path]:
    """Gera os 3 gráficos da análise exploratória e devolve os caminhos."""
    internos = df[~eh_interplanetario(df)]
    caminhos = []

    # 1) latência por módulo: mediana (ponto) e faixa de 10% a 90% (linha)
    estat = internos.groupby("modulo")["latencia_observada_ms"].describe(percentiles=[.1, .5, .9])
    estat = estat.sort_values("50%")
    fig, ax = g.nova_figura(8, 4.6)
    posicoes = np.arange(len(estat))
    ax.hlines(posicoes, estat["10%"], estat["90%"], color=g.AZUL, linewidth=2, alpha=0.35)
    ax.plot(estat["50%"], posicoes, "o", color=g.AZUL, markersize=8,
            markeredgecolor="white", markeredgewidth=1.5)
    for y, (mediana, p90) in enumerate(zip(estat["50%"], estat["90%"])):
        ax.text(p90 + 2, y, f"{mediana:.0f} ms", va="center", fontsize=8.5, color=g.TINTA_2)
    ax.set_yticks(posicoes)
    ax.set_yticklabels(estat.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Latência observada (ms)")
    ax.set_title(f"{estat.index[-1]} e {estat.index[-2]} têm as maiores latências internas")
    g.subtitulo(ax, "Ponto = mediana; linha = 80% centrais dos registros (P10 a P90), 30 sóis")
    caminhos.append(g.salvar_figura(fig, "eda_latencia_por_modulo.png"))

    # 2) latência x carga da rede, com a reta de tendência
    fig, ax = g.nova_figura(8, 4.6)
    x = internos["carga_rede_pct"].to_numpy()
    y = internos["latencia_observada_ms"].to_numpy()
    ax.scatter(x, y, s=22, color=g.AZUL, alpha=0.55, edgecolors="white", linewidths=0.6,
               label="Registros dos 9 módulos internos")
    coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), color=g.LARANJA, linewidth=2,
            label=f"Tendência linear: +{g.fmt(coef[0], 2)} ms por ponto de carga")
    ax.set_xlabel("Carga da rede (%)")
    ax.set_ylabel("Latência observada (ms)")
    ax.set_title("Mais carga na rede, mais latência")
    g.subtitulo(ax, "Cada ponto é um registro de um módulo em um sol")
    ax.legend(loc="upper left")
    caminhos.append(g.salvar_figura(fig, "eda_latencia_vs_carga.png"))

    # 3) correlação entre as medições (escala divergente azul-cinza-vermelho)
    from matplotlib.colors import LinearSegmentedColormap
    colunas = {"carga_rede_pct": "Carga", "qualidade_sinal_pct": "Qualidade",
               "perda_pacotes_pct": "Perda", "tensao_v": "Tensão", "corrente_a": "Corrente",
               "latencia_observada_ms": "Latência"}
    corr = internos[list(colunas)].rename(columns=colunas).corr()
    mapa = LinearSegmentedColormap.from_list("div", [g.AZUL, "#f0efec", g.LARANJA])
    fig, ax = g.nova_figura(6.6, 5.2)
    ax.grid(False)
    imagem = ax.imshow(corr.to_numpy(), cmap=mapa, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticks(range(len(corr)))
    ax.set_yticklabels(corr.index)
    for i in range(len(corr)):
        for j in range(len(corr)):
            valor = corr.iat[i, j]
            ax.text(j, i, g.fmt(valor, 2), ha="center", va="center", fontsize=8.5,
                    color="white" if abs(valor) > 0.6 else g.TINTA)
    for lado in ax.spines.values():
        lado.set_visible(False)
    barra = fig.colorbar(imagem, ax=ax, fraction=0.046, pad=0.04)
    barra.outline.set_visible(False)
    barra.ax.tick_params(labelsize=8, colors=g.TINTA_FRACA)
    barra.ax.yaxis.set_major_formatter(g.FormatadorBR())
    lat = corr["Latência"].drop("Latência")
    mais_forte = lat.abs().idxmax()
    ax.set_title(f"{mais_forte} é a medição mais ligada à latência (r = {g.fmt(lat[mais_forte], 2)})")
    g.subtitulo(ax, "Correlação de Pearson nos enlaces internos: laranja = positiva, azul = negativa")
    caminhos.append(g.salvar_figura(fig, "eda_correlacao.png"))
    return caminhos
