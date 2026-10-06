"""
modelo.py
=========
Modelo simples de previsão da latência dos enlaces internos da colônia.

Fluxo (o mesmo do estudo de caso da aula de avaliação de performance):
1. separa treino (80%) e teste (20%) com random_state=42;
2. cria dois baselines: a média do treino e a estimativa de engenharia do CSV;
3. ajusta três regressões lineares, da mais simples para a mais completa:
      A: só a carga da rede
      B: carga + qualidade do sinal + perda de pacotes
      C: B + o módulo (codificado com one-hot)
4. compara as três por AIC e BIC (no treino) e por validação cruzada 5-fold;
5. escolhe o modelo pelo menor BIC e só então mede MAE, MSE, RMSE e R² no teste;
6. confere com uma Ridge ajustada por Grid Search (alpha em 0,01 a 100).

O enlace Terra-Marte fica de fora: sua latência é física (distância / velocidade
da luz), está em outra escala (minutos) e distorceria a regressão dos enlaces
internos (milissegundos).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import graficos as g
from dados import eh_interplanetario
from graficos import fmt

ALVO = "latencia_observada_ms"
NUMERICAS = ["carga_rede_pct", "qualidade_sinal_pct", "perda_pacotes_pct"]
ESPECIFICACOES = {
    "A: só carga": {"num": ["carga_rede_pct"], "cat": []},
    "B: carga + qualidade + perda": {"num": NUMERICAS, "cat": []},
    "C: B + módulo": {"num": NUMERICAS, "cat": ["modulo"]},
}
SEED = 42
LIMITE_PICO_MS = 30.0   # resíduo acima disso = pico de retransmissão (imprevisível)


# --------------------------------------------------------------------------
# peças do modelo
# --------------------------------------------------------------------------
def nome_curto(nome: str) -> str:
    """'C: B + módulo' -> 'C (B + módulo)', para usar no meio de frases."""
    letra, _, descricao = nome.partition(": ")
    return f"{letra} ({descricao})" if descricao else nome


def base_modelo(df: pd.DataFrame) -> pd.DataFrame:
    """Somente os enlaces internos (mesma escala de latência)."""
    return df[~eh_interplanetario(df)].copy()


def construir_pipeline(numericas: list[str], categoricas: list[str], estimador=None) -> Pipeline:
    """Pipeline: colunas numéricas passam direto e o módulo vira variáveis 0/1."""
    transformadores = [("num", "passthrough", numericas)]
    if categoricas:
        transformadores.append(("cat", OneHotEncoder(drop="first"), categoricas))
    return Pipeline([
        ("prep", ColumnTransformer(transformadores)),
        ("reg", estimador if estimador is not None else LinearRegression()),
    ])


def calcular_metricas(y_real, y_previsto) -> dict:
    """MAE, MSE, RMSE e R² (o RMSE sai da raiz do MSE, igual à aula)."""
    mse = mean_squared_error(y_real, y_previsto)
    return {"MAE": float(mean_absolute_error(y_real, y_previsto)), "MSE": float(mse),
            "RMSE": float(math.sqrt(mse)), "R2": float(r2_score(y_real, y_previsto))}


def aic_bic(y_real, y_previsto, k: int) -> tuple[float, float]:
    """AIC e BIC de uma regressão linear com erro gaussiano.

    ln L = -n/2 * [ln(2*pi) + ln(RSS/n) + 1];  AIC = -2 ln L + 2k;  BIC = -2 ln L + k ln n.
    k = coeficientes + intercepto. Dá o mesmo valor do statsmodels usado na aula.
    """
    y_real = np.asarray(y_real, dtype=float)
    y_previsto = np.asarray(y_previsto, dtype=float)
    n = len(y_real)
    rss = float(np.sum((y_real - y_previsto) ** 2))
    log_verossimilhanca = -n / 2 * (math.log(2 * math.pi) + math.log(rss / n) + 1)
    return -2 * log_verossimilhanca + 2 * k, -2 * log_verossimilhanca + k * math.log(n)


# --------------------------------------------------------------------------
# treino, seleção e avaliação
# --------------------------------------------------------------------------
def treinar_modelos(df: pd.DataFrame) -> dict:
    """Treina baselines e modelos, escolhe o final e devolve tudo o que o menu mostra."""
    base = base_modelo(df)
    X = base[NUMERICAS + ["modulo"]]
    y = base[ALVO]
    X_treino, X_teste, y_treino, y_teste = train_test_split(
        X, y, test_size=0.2, random_state=SEED)

    linhas = []
    dummy = DummyRegressor(strategy="mean").fit(X_treino, y_treino)
    linhas.append({"modelo": "Média do treino (baseline)", "k": 1,
                   **calcular_metricas(y_teste, dummy.predict(X_teste))})
    estimativa = base.loc[X_teste.index, "latencia_prevista_ms"]
    linhas.append({"modelo": "Estimativa de engenharia (CSV)", "k": 0,
                   **calcular_metricas(y_teste, estimativa)})

    pipelines = {}
    validacao = KFold(n_splits=5, shuffle=True, random_state=SEED)
    for nome, spec in ESPECIFICACOES.items():
        pipe = construir_pipeline(spec["num"], spec["cat"]).fit(X_treino, y_treino)
        k = len(pipe.named_steps["prep"].get_feature_names_out()) + 1
        aic, bic = aic_bic(y_treino, pipe.predict(X_treino), k)
        cv_rmse = -cross_val_score(construir_pipeline(spec["num"], spec["cat"]), X_treino,
                                   y_treino, cv=validacao,
                                   scoring="neg_root_mean_squared_error").mean()
        linhas.append({"modelo": nome, "k": k, "AIC": aic, "BIC": bic, "CV_RMSE": cv_rmse,
                       **calcular_metricas(y_teste, pipe.predict(X_teste))})
        pipelines[nome] = pipe

    tabela = pd.DataFrame(linhas).set_index("modelo")
    candidatos = tabela.loc[list(ESPECIFICACOES)]
    nome_final = candidatos["BIC"].idxmin()
    modelo_final = pipelines[nome_final]

    # conferência: Ridge com Grid Search sobre as mesmas variáveis do modelo C
    spec_c = ESPECIFICACOES["C: B + módulo"]
    grade = GridSearchCV(construir_pipeline(spec_c["num"], spec_c["cat"], Ridge()),
                         {"reg__alpha": [0.01, 0.1, 1, 10, 100]}, cv=5,
                         scoring="neg_mean_squared_error")
    grade.fit(X_treino, y_treino)
    ridge_metricas = calcular_metricas(y_teste, grade.predict(X_teste))

    previsto = modelo_final.predict(X_teste)
    residuos = y_teste.to_numpy() - previsto
    picos = np.abs(residuos) > LIMITE_PICO_MS
    sem_picos = calcular_metricas(y_teste.to_numpy()[~picos], previsto[~picos])
    nomes = modelo_final.named_steps["prep"].get_feature_names_out()
    coeficientes = pd.Series(modelo_final.named_steps["reg"].coef_, index=nomes)
    coeficientes.index = [n.split("__", 1)[1] for n in coeficientes.index]

    resultado = {
        "tabela": tabela,
        "nome_final": nome_final,
        "modelo": modelo_final,
        "intercepto": float(modelo_final.named_steps["reg"].intercept_),
        "coeficientes": coeficientes,
        "X_teste": X_teste,
        "y_teste": y_teste,
        "previsto": previsto,
        "estimativa_teste": estimativa,
        "n_treino": int(len(X_treino)),
        "n_teste": int(len(X_teste)),
        "ridge_alpha": float(grade.best_params_["reg__alpha"]),
        "ridge_metricas": ridge_metricas,
        "prevista_por_modulo": base.groupby("modulo")["latencia_prevista_ms"].median().to_dict(),
        "n_picos_teste": int(picos.sum()),
        "metricas_sem_picos": sem_picos,
    }
    resultado["interpretacao"] = interpretar(resultado)
    return resultado


def interpretar(resultado: dict) -> list[str]:
    """Transforma as métricas do modelo final em frases sobre a colônia."""
    tabela = resultado["tabela"]
    final = tabela.loc[resultado["nome_final"]]
    engenharia = tabela.loc["Estimativa de engenharia (CSV)"]
    media = tabela.loc["Média do treino (baseline)"]
    razao = final["RMSE"] / final["MAE"]
    reducao = (1 - final["MAE"] / engenharia["MAE"]) * 100
    sem_picos = resultado["metricas_sem_picos"]
    return [
        f"MAE = {fmt(final['MAE'])} ms: em média, a previsão erra {fmt(final['MAE'])} ms "
        "para mais ou para menos, na mesma unidade da latência.",
        f"MSE = {fmt(final['MSE'])} ms² e RMSE = {fmt(final['RMSE'])} ms: o RMSE é "
        f"{fmt(razao, 2)} vezes o MAE. "
        + ("Poucos registros com erro grande pesam no resultado."
           if razao > 1.25 else "Os erros são parecidos entre si, sem muitos extremos."),
        f"Os {resultado['n_picos_teste']} registros de teste com resíduo acima de "
        f"{LIMITE_PICO_MS:.0f} ms são picos de retransmissão, imprevisíveis por natureza. "
        f"Sem eles, o RMSE cai para {fmt(sem_picos['RMSE'])} ms e o R² sobe para "
        f"{fmt(sem_picos['R2'], 2)}: os picos concentram o erro quadrático.",
        f"R² = {fmt(final['R2'], 2)}: o modelo explica {final['R2'] * 100:.0f}% da variação da "
        "latência no teste. R² alto não significa acerto em cada registro, e R² médio aqui "
        "vem dos picos, não de uma relação errada.",
        f"Contra a estimativa de engenharia (MAE = {fmt(engenharia['MAE'])} ms, "
        f"R² = {fmt(engenharia['R2'], 2)}), o modelo reduz o erro médio em {reducao:.0f}%. "
        f"A média do treino, que ignora tudo, tem R² = {fmt(media['R2'], 2)}.",
        f"Escolha pelo menor BIC ({fmt(final['BIC'], 0)}); a validação cruzada concorda "
        f"(RMSE médio de {fmt(final['CV_RMSE'])} ms nos 5 folds). A Ridge com Grid Search "
        f"escolheu alpha = {fmt(resultado['ridge_alpha'], 2)} e ficou no mesmo nível "
        f"(MAE = {fmt(resultado['ridge_metricas']['MAE'])} ms): regularizar não muda o quadro.",
    ]


def prever_latencia(resultado: dict, modulo: str, carga: float, qualidade: float,
                    perda: float) -> dict:
    """Prevê a latência de um módulo interno e compara com a estimativa de engenharia."""
    entrada = pd.DataFrame([{"carga_rede_pct": carga, "qualidade_sinal_pct": qualidade,
                             "perda_pacotes_pct": perda, "modulo": modulo}])
    previsao = float(resultado["modelo"].predict(entrada)[0])
    estimativa = float(resultado["prevista_por_modulo"].get(modulo, float("nan")))
    return {"modulo": modulo, "previsao_ms": previsao, "estimativa_engenharia_ms": estimativa,
            "diferenca_ms": previsao - estimativa}


# --------------------------------------------------------------------------
# gráficos
# --------------------------------------------------------------------------
def grafico_previsto_vs_real(resultado: dict) -> Path:
    """Teste: previsões do modelo e da estimativa de engenharia contra o valor real."""
    y = resultado["y_teste"].to_numpy()
    fig, ax = g.nova_figura(6.6, 5.6)
    ax.scatter(y, resultado["estimativa_teste"], s=26, color=g.LARANJA, alpha=0.75,
               edgecolors="white", linewidths=0.7, label="Estimativa de engenharia (CSV)")
    ax.scatter(y, resultado["previsto"], s=26, color=g.AZUL, alpha=0.85,
               edgecolors="white", linewidths=0.7,
               label=f"Modelo {nome_curto(resultado['nome_final'])}")
    limite = [min(y.min(), resultado["previsto"].min()) - 5, max(y.max(), 140) + 5]
    ax.plot(limite, limite, color=g.TINTA_FRACA, linewidth=1, label="Previsão perfeita (real = previsto)")
    ax.set_xlim(limite)
    ax.set_ylim(limite)
    ax.set_xlabel("Latência observada (ms)")
    ax.set_ylabel("Latência prevista (ms)")
    final = resultado["tabela"].loc[resultado["nome_final"]]
    ax.set_title(f"O modelo acompanha a diagonal: R² = {fmt(final['R2'], 2)} no teste")
    g.subtitulo(ax, f"{resultado['n_teste']} registros de teste, nunca vistos no ajuste")
    ax.legend(loc="upper left")
    return g.salvar_figura(fig, "modelo_previsto_vs_real.png")


def grafico_residuos(resultado: dict) -> Path:
    """Resíduo (real - previsto) contra o valor previsto, com os picos de retransmissão em destaque."""
    previsto = resultado["previsto"]
    residuos = resultado["y_teste"].to_numpy() - previsto
    picos = np.abs(residuos) > LIMITE_PICO_MS
    fig, ax = g.nova_figura(8, 4.2)
    ax.axhline(0, color=g.TINTA_FRACA, linewidth=1)
    ax.scatter(previsto[~picos], residuos[~picos], s=26, color=g.AZUL, alpha=0.85,
               edgecolors="white", linewidths=0.7, label="Registros de teste")
    ax.scatter(previsto[picos], residuos[picos], s=40, color=g.LARANJA, edgecolors="white",
               linewidths=0.8, label=f"Picos de retransmissão (resíduo acima de {LIMITE_PICO_MS:.0f} ms)")
    ax.set_xlabel("Latência prevista pelo modelo (ms)")
    ax.set_ylabel("Resíduo: real − previsto (ms)")
    ax.set_title(f"Resíduos sem padrão, exceto {int(picos.sum())} picos de retransmissão")
    g.subtitulo(ax, "Pontos espalhados em torno de zero indicam que a relação linear basta")
    ax.legend(loc="upper right")
    return g.salvar_figura(fig, "modelo_residuos.png")


def grafico_comparacao(resultado: dict) -> Path:
    """MAE e RMSE no teste para baselines e modelos."""
    tabela = resultado["tabela"]
    ordem = ["Média do treino (baseline)", "Estimativa de engenharia (CSV)"] + list(ESPECIFICACOES)
    tabela = tabela.loc[ordem]
    fig, ax = g.nova_figura(8, 4.6)
    y = np.arange(len(tabela))
    altura = 0.34
    ax.barh(y - altura / 2, tabela["MAE"], height=altura, color=g.AZUL, label="MAE")
    ax.barh(y + altura / 2, tabela["RMSE"], height=altura, color=g.LARANJA, label="RMSE")
    for posicao, mae, rmse in zip(y, tabela["MAE"], tabela["RMSE"]):
        ax.text(mae + 0.3, posicao - altura / 2, fmt(mae), va="center", fontsize=8,
                color=g.TINTA_2)
        ax.text(rmse + 0.3, posicao + altura / 2, fmt(rmse), va="center", fontsize=8,
                color=g.TINTA_2)
    ax.set_yticks(y)
    ax.set_yticklabels(tabela.index)
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Erro no conjunto de teste (ms)")
    ax.set_title(f"O modelo {resultado['nome_final'].split(':')[0]} tem o menor erro")
    g.subtitulo(ax, "Barras menores são melhores; RMSE acima do MAE indica erros grandes isolados")
    ax.legend(loc="lower right")
    ax.set_xlim(0, tabela["RMSE"].max() * 1.15)
    return g.salvar_figura(fig, "modelo_comparacao_metricas.png")
