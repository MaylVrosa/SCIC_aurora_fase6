"""
codigo_fonte.py  -  ARQUIVO PRINCIPAL
=====================================
SCIC - Sistema de Comunicação Interplanetária da Colônia Aurora Siger
FIAP - Ciência da Computação - Fase 6 - Atividade Integradora

Como executar (na pasta do projeto):
    python codigo_fonte.py           menu interativo
    python codigo_fonte.py --demo    roda as 10 opções em sequência, sem perguntas

O menu aceita o número da opção ou o começo de um comando: digitar "pri" abre a
priorização de alertas. Quem resolve o comando é uma trie (trie_busca.py).

Organização do código (cada arquivo cuida de uma parte do enunciado):
    dados.py               leitura, limpeza, consulta, cadastro e indicadores (5.1)
    erros.py               erro absoluto/relativo, ponto flutuante, Euler e Newton (5.2)
    modelo.py              regressão linear, baselines, métricas, AIC/BIC (5.3)
    heap_alertas.py        max-heap de alertas com heapify-up/down (5.4)
    trie_busca.py          trie para busca por prefixo (5.5)
    hardware.py            bases numéricas, códigos de sensor e eletricidade (5.6)
    gestao_inteligente.py  anomalias, eventos, manutenção preditiva, redundância (5.7)
    graficos.py            estilo dos gráficos e números no padrão brasileiro
    gerar_dados.py         gera a base simulada dados_aurora_siger.csv (seed 42)

As funções dos módulos só calculam e devolvem dados; quem imprime é este arquivo.
"""
from __future__ import annotations

import math
import sys
import textwrap
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")   # mantém o terminal limpo durante a demonstração

import pandas as pd  # noqa: E402

import dados as D  # noqa: E402
import erros as E  # noqa: E402
import gestao_inteligente as G  # noqa: E402
import graficos as GR  # noqa: E402
import hardware as HW  # noqa: E402
import heap_alertas as H  # noqa: E402
import modelo as M  # noqa: E402
import trie_busca as T  # noqa: E402
from graficos import fmt  # noqa: E402

LARGURA = 78
OPCOES = [
    (1, "Carregar ou cadastrar dados da colônia"),
    (2, "Consultar registros operacionais"),
    (3, "Indicadores e erros numéricos"),
    (4, "Executar uma previsão de latência"),
    (5, "Avaliar o desempenho do modelo"),
    (6, "Priorizar alertas (heap)"),
    (7, "Buscar por prefixo (trie)"),
    (8, "Dispositivos, bases numéricas e eletricidade"),
    (9, "Gerenciamento inteligente da comunicação"),
    (10, "Análise final dos resultados"),
    (0, "Sair"),
]
# primeira palavra dos nomes femininos, para escrever "da Estufa" e "do Centro Médico"
PALAVRAS_FEMININAS = {"comunicacao", "habitacao", "estufa", "antena"}

try:  # evita erro de acentuação em terminais antigos do Windows
    sys.stdout.reconfigure(errors="replace")
except (AttributeError, ValueError):
    pass

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 20)


# ==========================================================================
# estado da sessão
# ==========================================================================
class Sessao:
    """Estado do sistema durante a execução: a base carregada e resultados em cache."""

    def __init__(self, demo: bool = False) -> None:
        self.demo = demo
        self.df: pd.DataFrame | None = None
        self.relatorio_limpeza: dict = {}
        self.novos_nao_salvos = 0
        self.resultado_modelo: dict | None = None
        self.indice: dict | None = None
        self.bench_heap: pd.DataFrame | None = None
        self.bench_trie: pd.DataFrame | None = None
        self.trie_comandos = T.Trie()          # resolve "pri" -> opção 6 no menu
        for comando, opcao in T.COMANDOS_MENU.items():
            self.trie_comandos.inserir(comando, opcao)

    def preparar(self, recarregar: bool = False) -> pd.DataFrame:
        """Carrega o CSV (gerando-o se não existir) e calcula as colunas derivadas."""
        if self.df is None or recarregar:
            if not D.ARQUIVO_DADOS.exists():
                import gerar_dados
                gerar_dados.main()
            df, self.relatorio_limpeza = D.carregar_dados()
            self.novos_nao_salvos = 0
            self.atualizar(df)
        return self.df

    def atualizar(self, df: pd.DataFrame) -> None:
        """Recalcula potência e erros e descarta os resultados que dependem da base."""
        self.df = E.calcular_erros(HW.calcular_potencia(df))
        self.resultado_modelo = None
        self.indice = None

    def modelo(self) -> dict:
        if self.resultado_modelo is None:
            self.resultado_modelo = M.treinar_modelos(self.preparar())
        return self.resultado_modelo

    def indice_trie(self) -> dict:
        if self.indice is None:
            self.indice = T.construir_indice(self.preparar())
        return self.indice

    def benchmark_heap(self) -> pd.DataFrame:
        if self.bench_heap is None:
            self.bench_heap = H.benchmark_heap_vs_lista()
        return self.bench_heap

    def benchmark_trie(self) -> pd.DataFrame:
        if self.bench_trie is None:
            self.bench_trie = T.benchmark_trie_vs_linear()
        return self.bench_trie


# ==========================================================================
# entrada e saída
# ==========================================================================
def titulo(texto: str) -> None:
    print("\n" + "=" * LARGURA)
    print(f" {texto}")
    print("=" * LARGURA)


def secao(texto: str) -> None:
    print(f"\n-- {texto} " + "-" * max(3, LARGURA - len(texto) - 4))


def escrever(texto: str, inicio: str = "", recuo: str = "") -> None:
    """Imprime um parágrafo quebrando as linhas na largura do terminal."""
    print(textwrap.fill(texto, width=LARGURA, initial_indent=inicio, subsequent_indent=recuo))


def numero_curto(valor: float) -> str:
    """Número sem zeros à direita e sem ponto de milhar, fácil de redigitar: 1,5 / 80 / 1234,5."""
    texto = fmt(valor, 2).replace(".", "")
    return texto.rstrip("0").rstrip(",") if "," in texto else texto


def do_modulo(nome: str) -> str:
    """Contração com o artigo certo: 'da Estufa Hidropônica', 'do Centro Médico'."""
    feminino = D.normalizar_texto(nome).split()[0] in PALAVRAS_FEMININAS
    return ("da " if feminino else "do ") + nome


def perguntar(sessao: Sessao, texto: str, padrao: str = "") -> str:
    """Lê um texto; Enter aceita o padrão. No modo demo, usa o padrão direto."""
    rotulo = f"{texto} [{padrao}]: " if padrao != "" else f"{texto}: "
    if sessao.demo:
        print(rotulo + str(padrao))
        return str(padrao)
    resposta = input(rotulo).strip()
    return resposta if resposta else str(padrao)


def ler_numero(sessao: Sessao, texto: str, padrao: float, minimo: float | None = None,
               maximo: float | None = None, inteiro: bool = False) -> float:
    """Lê um número validando o tipo e a faixa; repete até receber um valor válido."""
    while True:
        resposta = perguntar(sessao, texto, numero_curto(padrao))
        try:
            valor = float(resposta.replace(",", "."))
            if not math.isfinite(valor):
                raise ValueError
            if inteiro:
                if valor != int(valor):
                    raise ValueError
                valor = int(valor)
        except ValueError:
            print("  Valor inválido: digite um número" + (" inteiro." if inteiro else "."))
            continue
        if (minimo is not None and valor < minimo) or (maximo is not None and valor > maximo):
            print(f"  Fora da faixa: use um valor de {numero_curto(minimo)} a {numero_curto(maximo)}.")
            continue
        return valor


def ler_sim_nao(sessao: Sessao, texto: str, padrao: bool = False) -> bool:
    while True:
        resposta = D.normalizar_texto(perguntar(sessao, texto + " (s/n)", "s" if padrao else "n"))
        if resposta in ("s", "sim"):
            return True
        if resposta in ("n", "nao"):
            return False
        print("  Responda com s ou n.")


def escolher_modulo(sessao: Sessao, texto: str, padrao: str, permitidos: list[str]) -> str:
    """Escolha de módulo por prefixo, resolvida pela trie de módulos ('hab' -> 2 opções)."""
    trie_modulos = sessao.indice_trie()["módulos"]
    while True:
        digitado = perguntar(sessao, texto, padrao)
        achados = sorted({ref for _, refs in trie_modulos.buscar_prefixo(digitado) for ref in refs
                          if ref in permitidos}, key=D.normalizar_texto)
        if len(achados) == 1:
            print(f"  Módulo escolhido: {achados[0]}")
            return achados[0]
        if achados:
            print("  Mais de um módulo começa assim: " + ", ".join(achados))
        else:
            print("  Nenhum módulo encontrado. Opções: " + ", ".join(permitidos))


def mostrar_tabela(tabela: pd.DataFrame, formatos: dict | None = None, indice: bool = True) -> None:
    """Imprime uma tabela com números no padrão brasileiro."""
    tabela = tabela.copy()
    for coluna, casas in (formatos or {}).items():
        if coluna in tabela.columns:
            tabela[coluna] = [fmt(v, casas) for v in tabela[coluna]]
    print(tabela.to_string(index=indice))


def informar_graficos(sessao: Sessao, caminhos: list[Path]) -> None:
    """Lista os gráficos salvos e oferece abrir no visualizador do sistema."""
    print("\nGráficos salvos em graficos_ou_imagens/:")
    for caminho in caminhos:
        print(f"  - {caminho.name}")
    if not sessao.demo and ler_sim_nao(sessao, "Abrir os gráficos agora?", False):
        for caminho in caminhos:
            if not GR.abrir_arquivo(caminho):
                print(f"  Não foi possível abrir {caminho.name}; abra pela pasta.")


def latencia_legivel(ms: float) -> str:
    """Enlaces internos em milissegundos; enlace Terra-Marte em minutos."""
    return f"{fmt(ms / 60000, 2)} min" if ms > 60000 else f"{fmt(ms)} ms"


# ==========================================================================
# 1 - carregar ou cadastrar
# ==========================================================================
def opcao_carregar(sessao: Sessao) -> None:
    titulo("1 - Carregar ou cadastrar dados da colônia")
    recarregar = sessao.novos_nao_salvos == 0      # não descarta cadastros ainda não salvos
    df = sessao.preparar(recarregar=recarregar)
    r = sessao.relatorio_limpeza
    print(f"Arquivo: {D.ARQUIVO_DADOS.name}")
    print(f"Linhas lidas: {r['linhas_lidas']}  |  linhas válidas após a limpeza: {r['linhas_finais']}")
    print(f"  - duplicadas removidas: {r['duplicadas_removidas']}")
    print(f"  - sem latência observada (removidas): {r['sem_latencia_removidas']}")
    imputados = ", ".join(f"{c} ({n})" for c, n in r["valores_imputados"].items()) or "nenhum"
    print(f"  - vazios completados com a mediana do módulo: {imputados}")
    print(f"  - percentuais fora de 0 a 100 corrigidos: {r['percentuais_corrigidos']}")
    print(f"Módulos: {df['modulo'].nunique()}  |  sóis: {df['ciclo'].nunique()}  |  "
          f"status: " + ", ".join(f"{s} {n}" for s, n in df["status"].value_counts().items()))
    if not recarregar:
        print(f"Base em memória mantida: {sessao.novos_nao_salvos} registro(s) cadastrado(s) "
              "nesta sessão ainda não foram salvos no CSV.")

    secao("Estrutura em Python: DataFrame do Pandas")
    extras = len(df.columns) - len(D.COLUNAS_OBRIGATORIAS)
    escrever(f"{len(df)} linhas e {len(df.columns)} colunas: as {len(D.COLUNAS_OBRIGATORIAS)} do CSV "
             f"e {extras} calculadas pelo sistema (potência, consumo, queda de tensão e erros).")
    amostra = df[["id_registro", "ciclo", "modulo", "codigo_sensor", "latencia_prevista_ms",
                  "latencia_observada_ms", "status"]].head(5)
    mostrar_tabela(amostra, {"latencia_prevista_ms": 1, "latencia_observada_ms": 1}, indice=False)

    if sessao.demo:
        secao("Cadastro de um novo registro (demonstração)")
        cadastrar(sessao)
        sessao.preparar(recarregar=True)
        print(f"Registro de demonstração descartado: a base volta aos {len(sessao.df)} registros do CSV.")
        return
    if ler_sim_nao(sessao, "\nDeseja cadastrar um novo registro?", False):
        cadastrar(sessao)


def cadastrar(sessao: Sessao) -> None:
    """Pede os campos medidos; tipo, código, distância e prioridade vêm do próprio módulo."""
    df = sessao.preparar()
    modulos = sorted(df["modulo"].unique(), key=D.normalizar_texto)
    modulo = escolher_modulo(sessao, "Módulo (pode digitar só o começo)", "Centro Médico", modulos)
    historico = df[df["modulo"] == modulo]
    referencia = historico.iloc[-1]
    registro = {
        "modulo": modulo,
        "ciclo": ler_numero(sessao, "Sol (ciclo)", int(df["ciclo"].max()) + 1, 1, 10_000, True),
        "carga_rede_pct": ler_numero(sessao, "Carga da rede (%)", 60, 0, 100),
        "qualidade_sinal_pct": ler_numero(sessao, "Qualidade do sinal (%)", 85, 0, 100),
        "perda_pacotes_pct": ler_numero(sessao, "Perda de pacotes (%)", 1.0, 0, 100),
        "tensao_v": ler_numero(sessao, "Tensão do transmissor (V)", 28.0, 0.1, 60),
        "corrente_a": ler_numero(sessao, "Corrente do transmissor (A)",
                                 round(float(historico["corrente_a"].median()), 2), 0, 50),
        "latencia_prevista_ms": float(referencia["latencia_prevista_ms"]),
        "latencia_observada_ms": ler_numero(sessao, "Latência observada (ms)",
                                            round(float(historico["latencia_observada_ms"].median()), 1),
                                            0.1, 5e6),
    }
    # mesma regra de status usada na base: tensão, sinal, perda e latência acima do previsto
    em_alerta = (registro["tensao_v"] < HW.LIMITE_QUEDA_TENSAO_V
                 or registro["qualidade_sinal_pct"] < 70 or registro["perda_pacotes_pct"] > 4
                 or registro["latencia_observada_ms"] > 1.35 * registro["latencia_prevista_ms"])
    sugerido = "alerta" if em_alerta else "ativo"
    while True:
        status = D.normalizar_texto(perguntar(sessao, "Status (ativo/manutenção/alerta)", sugerido))
        status = {"manutencao": "manutenção"}.get(status, status)
        if status in D.STATUS_VALIDOS:
            break
        print("  Status inválido: use ativo, manutenção ou alerta.")
    registro["status"] = status
    registro["mensagem_alerta"] = perguntar(sessao, "Mensagem do alerta (opcional)",
                                            "Latência acima do previsto" if status == "alerta" else "")
    try:
        novo_df = D.cadastrar_registro(sessao.df, registro)
    except ValueError as erro:
        print(f"Cadastro recusado: {erro}")
        return
    sessao.atualizar(novo_df)
    novo = sessao.df[sessao.df["id_registro"] == sessao.df["id_registro"].max()].iloc[0]
    print(f"\nRegistro {novo['id_registro']} cadastrado: {novo['modulo']}, sol {novo['ciclo']}, "
          f"status {novo['status']}.")
    print(f"Herdados do módulo: tipo {novo['tipo_modulo']}, código {novo['codigo_sensor']}, "
          f"prioridade {novo['prioridade_base']}.")
    print(f"Calculados: potência {fmt(novo['potencia_w'])} W; erro da estimativa de engenharia "
          f"{fmt(novo['erro_rel_pct'])}% ({novo['faixa_erro']}).")
    if ler_sim_nao(sessao, "Salvar também no arquivo CSV?", False):
        D.salvar_dados(sessao.df)
        sessao.novos_nao_salvos = 0
        print("Base salva. Para voltar à base original, rode: python gerar_dados.py")
    else:
        sessao.novos_nao_salvos += 1
        print("Registro mantido só na memória desta sessão.")


# ==========================================================================
# 2 - consultar
# ==========================================================================
def opcao_consultar(sessao: Sessao) -> None:
    titulo("2 - Consultar registros operacionais")
    df = sessao.preparar()
    print("Deixe em branco para não filtrar. Acentos e maiúsculas não importam.")
    modulo = perguntar(sessao, "Módulo (nome ou parte do nome)", "medico")
    status = perguntar(sessao, "Status (ativo, manutenção ou alerta)", "alerta")
    ciclo_txt = perguntar(sessao, "Sol (número)", "")
    ciclo = None
    if ciclo_txt:
        try:
            ciclo = int(ciclo_txt)
        except ValueError:
            print("  Sol inválido: o filtro de sol foi ignorado.")
    resultado = D.consultar_registros(df, modulo or None, status or None, ciclo)
    print(f"\n{len(resultado)} registro(s) encontrado(s).")
    if resultado.empty:
        return
    tabela = resultado[["id_registro", "ciclo", "modulo", "status", "latencia_observada_ms",
                        "erro_rel_pct", "tensao_v", "mensagem_alerta"]].rename(columns={
        "id_registro": "id", "ciclo": "sol", "modulo": "módulo", "latencia_observada_ms": "latência ms",
        "erro_rel_pct": "erro rel. %", "tensao_v": "tensão V", "mensagem_alerta": "mensagem"})
    mostrar_tabela(tabela.head(15), {"latência ms": 1, "erro rel. %": 1, "tensão V": 2}, indice=False)
    if len(resultado) > 15:
        print(f"... e mais {len(resultado) - 15} registro(s).")


# ==========================================================================
# 3 - indicadores e erros numéricos
# ==========================================================================
def opcao_indicadores_erros(sessao: Sessao) -> None:
    titulo("3 - Indicadores e erros numéricos")
    print("1 Indicadores de comunicação    2 Erro absoluto e relativo")
    print("3 Ponto flutuante (IEEE 754)     4 Euler e Newton    5 Tudo")
    escolha = int(ler_numero(sessao, "Parte", 5, 1, 5, True))
    partes = [1, 2, 3, 4] if escolha == 5 else [escolha]
    if 1 in partes:
        mostrar_indicadores(sessao)
    if 2 in partes:
        mostrar_erros(sessao)
    if 3 in partes:
        mostrar_ponto_flutuante(sessao)
    if 4 in partes:
        mostrar_euler(sessao)


def mostrar_indicadores(sessao: Sessao) -> None:
    df = sessao.preparar()
    tabela, resumo = D.calcular_indicadores(df)
    secao("Indicadores de comunicação por módulo")
    exibicao = pd.DataFrame({
        "latência média": [latencia_legivel(v) for v in tabela["latencia_media_ms"]],
        "qualidade %": tabela["qualidade_media_pct"],
        "perda %": tabela["perda_media_pct"],
        "disponível %": tabela["disponibilidade_pct"],
        "alertas": tabela["alertas"],
        "potência W": tabela["potencia_media_w"],
    }, index=tabela.index)
    exibicao.index.name = "módulo"
    mostrar_tabela(exibicao, {"qualidade %": 1, "perda %": 2, "disponível %": 1, "potência W": 1})
    print("Disponível % = sóis com status ativo (sem alerta e sem manutenção).")
    secao("Visão da colônia")
    print(f"Registros: {resumo['registros']}  |  em alerta: {fmt(resumo['pct_alerta'])}%  |  "
          f"disponibilidade média: {fmt(resumo['disponibilidade_media_pct'])}%")
    print(f"Latência média dos enlaces internos: {fmt(resumo['latencia_media_interna_ms'])} ms")
    print(f"Enlace Terra-Marte: de {fmt(resumo['latencia_min_terra_min'], 2)} a "
          f"{fmt(resumo['latencia_max_terra_min'], 2)} min (225 a 240 milhões de km neste período)")
    nome, valor = resumo["pior_disponibilidade"]
    print(f"Menor disponibilidade: {nome} ({fmt(valor)}% dos sóis com status ativo)")
    informar_graficos(sessao, D.gerar_graficos_exploratorios(df))


def mostrar_erros(sessao: Sessao) -> None:
    df = sessao.preparar()
    secao("Erro absoluto e erro relativo (estimativa de engenharia x observado)")
    print("erro absoluto = |prevista - observada|            (em ms)")
    print("erro relativo = erro absoluto / |observada| x 100  (em %)")
    print("Faixas do erro relativo: aceitável até 10%, atenção até 25%, crítico acima.")
    print("Módulos que mantêm vidas (prioridade 5) usam faixas mais rígidas: 5% e 15%.")
    tabela = E.resumo_erros_por_modulo(df)
    exibicao = pd.DataFrame({
        "erro abs. médio": [latencia_legivel(v) for v in tabela["erro_abs_medio_ms"]],
        "erro rel. médio %": tabela["erro_rel_medio_pct"],
        "erro rel. máx. %": tabela["erro_rel_max_pct"],
        "% críticos": tabela["criticos_pct"],
        "faixa": tabela["faixa_media"],
    }, index=tabela.index)
    exibicao.index.name = "módulo"
    mostrar_tabela(exibicao, {"erro rel. médio %": 2, "erro rel. máx. %": 1, "% críticos": 0})
    escala = E.comparar_escalas(df)
    secao("Por que o erro relativo é necessário")
    print(f"Enlaces internos: erro absoluto médio de {fmt(escala['interno_erro_abs_ms'])} ms; "
          f"erro relativo médio de {fmt(escala['interno_erro_rel_pct'])}%.")
    print(f"Enlace Terra-Marte: erro absoluto médio de {fmt(escala['terra_erro_abs_ms'], 0)} ms; "
          f"erro relativo médio de {fmt(escala['terra_erro_rel_pct'], 2)}%.")
    vezes_abs = escala["terra_erro_abs_ms"] / escala["interno_erro_abs_ms"]
    vezes_rel = escala["interno_erro_rel_pct"] / escala["terra_erro_rel_pct"]
    escrever(f"O erro absoluto da Terra é {vezes_abs:.0f} vezes maior, mas o relativo é "
             f"{vezes_rel:.0f} vezes menor: só o erro relativo compara módulos de escalas diferentes.")
    escrever(f"Viés: nos enlaces internos a estimativa fica {fmt(escala['vies_interno_ms'])} ms abaixo "
             "do real, em média, porque usa a carga planejada (55%) e ignora a carga real e a perda "
             "de pacotes.")
    informar_graficos(sessao, [E.grafico_erros_relativos(df)])


def mostrar_ponto_flutuante(sessao: Sessao) -> None:
    pf = E.demonstrar_ponto_flutuante()
    secao("Ponto flutuante (IEEE 754, precisão dupla)")
    print(f"0.1 + 0.2              = {pf['soma_01_02']!r}")
    print(f"0.1 + 0.2 == 0.3       -> {pf['soma_igual_03']}   |   math.isclose(0.1 + 0.2, 0.3) -> "
          f"{pf['isclose_03']}")
    print(f"28 V x 1.6 A           = {pf['potencia_28x16']!r} W   (exibido: "
          f"{fmt(pf['potencia_arredondada'], 2)} W)")
    print(f"(1e16 + 1) - 1e16      = {pf['grande_mais_1']}   (o 1 some: faltam dígitos nessa escala)")
    print(f"épsilon da máquina     = {pf['epsilon']:.3e}")
    print(f"0.1 é guardado com erro absoluto de {pf['erro_abs_de_01']:.2e} "
          f"(relativo de {pf['erro_rel_de_01']:.2e})")
    ieee = pf["ieee_latencia"]
    print(f"\nLatência de {ieee['valor']} ms em 64 bits:")
    print(f"  sinal {ieee['sinal']} | expoente {ieee['expoente']} (2^{ieee['expoente_real']})")
    print(f"  mantissa {ieee['mantissa']}")
    print(f"  valor realmente guardado: {ieee['valor_exato_guardado']}")
    arr = E.propagacao_arredondamento(sessao.preparar())
    secao("Arredondamento na base: propagação para P = V x I")
    escrever(f"O CSV guarda tensão e corrente com 2 casas (erro de até {fmt(arr['meia_unidade'], 3)}). "
             f"No módulo de maior potência, {arr['modulo']} ({fmt(arr['tensao_v'], 2)} V x "
             f"{fmt(arr['corrente_a'], 2)} A = {fmt(arr['potencia_w'])} W), o erro máximo de P é "
             f"I x 0,005 + V x 0,005 = {fmt(arr['erro_max_w'], 2)} W, só "
             f"{fmt(arr['erro_rel_pct'], 2)}% da potência: não muda nenhuma decisão.")
    print("Conclusão: comparar floats com tolerância e arredondar só na hora de exibir.")


def mostrar_euler(sessao: Sessao) -> None:
    df = sessao.preparar()
    res = E.analise_euler(df)
    secao(f"Euler: queda da qualidade do sinal {do_modulo(res['modulo'])}")
    print(f"Modelo: dQ/dt = -taxa x Q, com taxa = {fmt(res['taxa_por_sol'], 4)} por sol e "
          f"Q0 = {fmt(res['q0'])}%")
    print("        (ajustados aos dados da base; t = 0 no sol 1)")
    print("Euler:  Q(n+1) = Q(n) + h x (-taxa x Q(n))")
    print("Exata:  Q(t) = Q0 x e^(-taxa x t), comparada com Euler no sol 29 (t = 28)")
    tabela = res["tabela"].rename(columns={"passo_h": "passo h (sóis)", "euler_final": "Euler",
                                           "exato_final": "exato", "erro_abs": "erro abs.",
                                           "erro_rel_pct": "erro rel. %"})
    mostrar_tabela(tabela, {"passo h (sóis)": 1, "Euler": 4, "exato": 4, "erro abs.": 4,
                            "erro rel. %": 3}, indice=False)
    print("Passo pela metade, erro pela metade: Euler é um método de 1ª ordem.")
    secao("Newton: em que sol a qualidade cruza 70%?")
    print(f"f(t) = Q0 x e^(-taxa x t) - 70 = 0  ->  t = {fmt(res['t_cruzamento_newton'], 4)} "
          f"em {res['iteracoes_newton']} iterações")
    print(f"Solução analítica t = ln(Q0/70) / taxa = {fmt(res['t_cruzamento_analitico'], 4)} (confere)")
    print(f"Previsão: o sinal cruza 70% por volta do sol {fmt(res['sol_cruzamento'])}.")
    informar_graficos(sessao, [E.grafico_euler(df, res)])


# ==========================================================================
# 4 e 5 - modelo
# ==========================================================================
def opcao_previsao(sessao: Sessao) -> None:
    titulo("4 - Executar uma previsão de latência")
    resultado = sessao.modelo()
    escrever(f"Modelo em uso: {M.nome_curto(resultado['nome_final'])}, regressão linear treinada com "
             f"{resultado['n_treino']} registros dos enlaces internos.")
    coef = resultado["coeficientes"]
    print("Como o modelo pensa (coeficientes):")
    print(f"  +1 ponto de carga da rede      -> {fmt(coef['carga_rede_pct'], 2)} ms")
    print(f"  +1 ponto de qualidade do sinal -> {fmt(coef['qualidade_sinal_pct'], 2)} ms")
    print(f"  +1 ponto de perda de pacotes   -> {fmt(coef['perda_pacotes_pct'], 2)} ms")
    print("  + um ajuste fixo para cada módulo (distância e equipamento)")
    internos = sorted(resultado["prevista_por_modulo"], key=D.normalizar_texto)
    print()
    modulo = escolher_modulo(sessao, "Módulo (pode digitar só o começo)", "Centro Médico", internos)
    carga = ler_numero(sessao, "Carga da rede (%)", 80, 0, 100)
    qualidade = ler_numero(sessao, "Qualidade do sinal (%)", 85, 0, 100)
    perda = ler_numero(sessao, "Perda de pacotes (%)", 1.5, 0, 100)
    p = M.prever_latencia(resultado, modulo, carga, qualidade, perda)
    secao("Resultado")
    print(f"{modulo}: latência prevista pelo modelo = {fmt(p['previsao_ms'])} ms")
    print(f"Estimativa de engenharia para o módulo  = {fmt(p['estimativa_engenharia_ms'])} ms")
    if p["diferenca_ms"] > 5:
        escrever(f"Diferença de {fmt(p['diferenca_ms'])} ms: com a rede a {fmt(carga, 0)}% de carga, "
                 "a estimativa fixa (feita para 55%) subestima a latência.")
    elif p["diferenca_ms"] < -5:
        escrever(f"Diferença de {fmt(p['diferenca_ms'])} ms: com a rede a {fmt(carga, 0)}% de carga, "
                 "a estimativa fixa (feita para 55%) superestima a latência.")
    else:
        print(f"Diferença de {fmt(p['diferenca_ms'])} ms: as duas estimativas concordam.")


def opcao_avaliacao(sessao: Sessao) -> None:
    titulo("5 - Avaliar o desempenho do modelo")
    resultado = sessao.modelo()
    escrever(f"Divisão: {resultado['n_treino']} registros de treino e {resultado['n_teste']} de teste "
             "(80/20, random_state=42). O enlace Terra-Marte fica fora: outra escala de latência.")
    tabela = resultado["tabela"][["k", "MAE", "MSE", "RMSE", "R2", "AIC", "BIC", "CV_RMSE"]]
    secao("Métricas no teste (AIC, BIC e validação cruzada calculados só no treino)")
    mostrar_tabela(tabela, {"MAE": 2, "MSE": 1, "RMSE": 2, "R2": 3, "AIC": 1, "BIC": 1,
                            "CV_RMSE": 2})
    print("k = parâmetros estimados; MAE, RMSE e CV_RMSE em ms; MSE em ms².")
    print(f"\nModelo escolhido: {M.nome_curto(resultado['nome_final'])}, pelo menor BIC.")
    secao("Interpretação")
    for frase in resultado["interpretacao"]:
        escrever(frase, inicio="- ", recuo="  ")
    informar_graficos(sessao, [M.grafico_comparacao(resultado),
                               M.grafico_previsto_vs_real(resultado),
                               M.grafico_residuos(resultado)])


# ==========================================================================
# 6 - heap
# ==========================================================================
def opcao_heap(sessao: Sessao) -> None:
    titulo("6 - Priorizar alertas (max-heap)")
    df = sessao.preparar()
    janela = int(ler_numero(sessao, "Considerar alertas dos últimos quantos sóis", H.JANELA_SOIS,
                            1, 30, True))
    heap = H.montar_fila_alertas(df, janela)
    if len(heap) == 0:
        print("Nenhum alerta aberto nessa janela.")
        return
    print("\nCritério de prioridade (score):")
    print("  3 x prioridade do módulo (5 = vida humana) + erro relativo / 10 (teto de 50%)")
    print("  + 2 se houver queda de tensão + 1 se qualidade < 70% + 1 se perda > 4%")
    print("  + 0,2 por sol de espera (quanto mais antigo, mais urgente)")
    print(f"\n{len(heap)} alertas inseridos um a um, cada um com heapify-up.")

    secao("Como cada alerta é representado")
    score, ordem, alerta = heap.item_topo()
    print("Tupla (score, ordem de chegada, dicionário). A ordem desempata scores iguais.")
    print(f"Exemplo, a raiz do heap: ({score}, {ordem}, {{")
    for chave in ("id_registro", "ciclo", "modulo", "codigo_sensor", "mensagem", "erro_rel_pct",
                  "tensao_v"):
        print(f"    '{chave}': {alerta[chave]!r},")
    print("})")
    c = alerta["componentes"]
    print(f"Score = módulo {fmt(c['modulo'], 0)} + erro {fmt(c['erro_previsao'], 2)} + tensão "
          f"{fmt(c['tensao'], 0)} + sinal {fmt(c['sinal'], 0)} + espera {fmt(c['espera'], 1)} "
          f"= {fmt(score, 2)}")

    secao("Como o heap guarda os alertas: uma lista que representa uma árvore")
    vetor = heap.vetor()
    nivel, inicio = 0, 0
    while inicio < len(vetor) and nivel < 4:
        fim = min(len(vetor), 2 ** (nivel + 1) - 1)
        print(f"  nível {nivel}: " + "  ".join(fmt(v, 2) for v in vetor[inicio:fim]))
        nivel, inicio = nivel + 1, fim
    if inicio < len(vetor):
        print(f"  ... mais {len(vetor) - inicio} alertas nos níveis de baixo")
    print("  pai de i = (i-1)//2; filhos de i = 2i+1 e 2i+2")
    print(f"  todo pai tem score maior ou igual ao dos filhos: {heap.valido()}")

    secao("Os 5 alertas mais urgentes (cada extração faz heapify-down)")
    primeiro = None
    for posicao in range(1, min(5, len(heap)) + 1):
        score, alerta = heap.extrair_mais_urgente()
        primeiro = primeiro or (score, alerta)
        print(f"{posicao}. score {fmt(score, 2)} | sol {alerta['ciclo']} | {alerta['modulo']} "
              f"({alerta['codigo_sensor']}) | {alerta['mensagem']}")
    print(f"\nConferência com o heapq do Python (mesmo top 5): {H.conferir_com_heapq(df, 5, janela)}")

    secao("Automação simulada: JSON que iria para um webhook do n8n")
    print(H.payload_webhook(*primeiro))

    secao("Por que heap e não lista simples? (tempo medido)")
    print("Medindo...", end=" ", flush=True)
    tabela = sessao.benchmark_heap()
    print("pronto.")
    mostrar_tabela(tabela.rename(columns={"n_alertas": "alertas", "lista_ms": "lista (ms)",
                                          "heap_ms": "heap (ms)", "vantagem_x": "heap mais rápido (x)"}),
                   {"lista (ms)": 1, "heap (ms)": 1, "heap mais rápido (x)": 1}, indice=False)
    escrever("Lista: achar o maior exige olhar todos os alertas, O(n) a cada consulta. Heap: inserir "
             "custa O(log n) e o mais urgente está sempre na posição 0, O(1).")
    informar_graficos(sessao, [H.grafico_benchmark(tabela)])


# ==========================================================================
# 7 - trie
# ==========================================================================
def mostrar_busca(indice: dict, prefixo: str) -> None:
    resultados = T.buscar_em_tudo(indice, prefixo)
    if not resultados:
        print(f"  Nada começa com '{prefixo}'.")
        return
    for categoria, achados in resultados.items():
        if categoria == "palavras de alerta":
            textos = [f"{termo} ({len(refs)} registros)" for termo, refs in achados]
        elif categoria == "códigos de sensor":
            textos = [f"{termo} = {refs[0]}" for termo, refs in achados]
        elif categoria == "comandos do menu":
            textos = [f"{termo} -> opção {refs[0]}" for termo, refs in achados]
        else:
            textos = [termo for termo, _ in achados]
        escrever(f"  {categoria}: " + "; ".join(textos), recuo="    ")


def opcao_trie(sessao: Sessao) -> None:
    titulo("7 - Buscar por prefixo (trie)")
    indice = sessao.indice_trie()
    for categoria, trie in indice.items():
        print(f"Trie de {categoria}: {trie.total_termos} termos em {trie.contar_nos()} nós")
    escrever("O texto é normalizado: maiúsculas e acentos não importam ('comunicacao' acha "
             "'Comunicação'). Buscar um prefixo de m letras custa O(m), seja qual for o tamanho da base.")
    if sessao.demo:
        for prefixo in ["co", "com", "0x1", "lat"]:
            print(f"\nPrefixo: {prefixo}")
            mostrar_busca(indice, prefixo)
        print()
        escrever("Repare: 'co' traz Comando, Comunicação e Controle; 'com' não traz Controle "
                 "Ambiental, que começa com 'con'. A trie devolve só o que tem o prefixo exato.")
    else:
        while True:
            prefixo = input("\nPrefixo para buscar (Enter para terminar): ").strip()
            if not prefixo:
                break
            mostrar_busca(indice, prefixo)
    exata = indice["códigos de sensor"].buscar("0x5101")
    print(f"\nBusca exata, como na aula: '0x5101' -> {exata[0] if exata else 'não encontrado'}")

    secao("Por que trie e não busca linear? (tempo medido)")
    print("Medindo...", end=" ", flush=True)
    tabela = sessao.benchmark_trie()
    print("pronto.")
    mostrar_tabela(tabela.rename(columns={"n_codigos": "códigos", "linear_ms": "linear (ms)",
                                          "trie_ms": "trie (ms)", "vantagem_x": "trie mais rápida (x)"}),
                   {"códigos": 0, "linear (ms)": 3, "trie (ms)": 4, "trie mais rápida (x)": 0},
                   indice=False)
    escrever("Busca linear: compara o prefixo com todos os códigos, O(n x m). Trie: desce m níveis e "
             "percorre só a subárvore do prefixo, O(m + k), com k = resultados.")
    informar_graficos(sessao, [T.grafico_benchmark(tabela)])


# ==========================================================================
# 8 - hardware
# ==========================================================================
def opcao_hardware(sessao: Sessao) -> None:
    titulo("8 - Dispositivos, bases numéricas e eletricidade")
    df = sessao.preparar()
    secao("Código do sensor: hexadecimal -> binário -> decimal")
    print("Formato de 16 bits: 4 bits de tipo | 4 bits de módulo | 8 bits de sensor")
    while True:
        codigo = perguntar(sessao, "Código do sensor", "0x1201")
        try:
            info = HW.decodificar_codigo(codigo)
            break
        except ValueError as erro:
            print(f"  {erro}")
    valor = info["decimal"]
    expoentes = HW.potencias_de_dois(valor)
    print(f"{info['codigo']} em binário (cada dígito hex vira 4 bits): {info['binario']}")
    if expoentes:
        print("em decimal (soma de potências de 2): " + " + ".join(f"2^{e}" for e in expoentes)
              + " = " + " + ".join(str(2 ** e) for e in expoentes) + f" = {valor}")
    else:
        print("em decimal: 0")
    print(f"de volta a hexadecimal (divisões sucessivas por 16): 0x{HW.decimal_para_hexadecimal(valor)}")
    print("campos separados com operações de bits:")
    print(f"  tipo   = ({valor} >> 12) & 0xF = {info['tipo']} ({info['tipo_nome']})")
    print(f"  módulo = ({valor} >> 8) & 0xF  = {info['modulo']}")
    print(f"  sensor = {valor} & 0xFF        = {info['sensor']}")
    print(f"conferido com int(), bin() e hex() do Python: {info['confere_python']}")
    modulos = df[df["codigo_sensor"] == info["codigo"]]["modulo"].unique()
    print(f"módulo na base: {modulos[0] if len(modulos) else 'nenhum com esse código'}")

    secao("Eletricidade: P = V x I e consumo por sol (1 sol = 24,66 h)")
    tabela = HW.resumo_eletrico(df)
    tabela.index.name = "módulo"
    mostrar_tabela(tabela.rename(columns={"tensao_media_v": "tensão V", "corrente_media_a": "corrente A",
                                          "potencia_media_w": "potência W",
                                          "consumo_30_sois_kwh": "kWh em 30 sóis",
                                          "quedas_tensao": "quedas"}),
                   {"tensão V": 2, "corrente A": 2, "potência W": 1, "kWh em 30 sóis": 1})
    quedas = df[df["queda_tensao"]]
    print(f"\nQuedas de tensão (abaixo de {fmt(HW.LIMITE_QUEDA_TENSAO_V)} V, 90% dos 28 V nominais):")
    for _, linha in quedas.iterrows():
        print(f"  sol {linha['ciclo']}: {linha['modulo']} com {fmt(linha['tensao_v'], 2)} V "
              f"-> potência de {fmt(linha['potencia_w'])} W")

    secao("Lei de Ohm (V = R x I) no painel do centro de controle")
    led = HW.resistor_led()
    queda_led = led["tensao_fonte_v"] - led["tensao_led_v"]
    print(f"LED de alerta (fonte de {fmt(led['tensao_fonte_v'])} V, LED de {fmt(led['tensao_led_v'])} V, "
          f"corrente de {fmt(led['corrente_a'] * 1000, 0)} mA):")
    print(f"  R = ({fmt(led['tensao_fonte_v'])} - {fmt(led['tensao_led_v'])}) / "
          f"{fmt(led['corrente_a'], 2)} = {fmt(led['resistencia_ohm'], 0)} ohms;  "
          f"P = V x I = {fmt(queda_led)} x {fmt(led['corrente_a'], 2)} = {fmt(led['potencia_w'], 2)} W")
    print(f"Associação em paralelo: 1/R = 1/300 + 1/300  ->  R = "
          f"{fmt(HW.associacao_resistores([300, 300], 'paralelo'), 0)} ohms (mesmo resistor do LED)")
    shunt = HW.sensor_shunt()
    print(f"Sensor de corrente (shunt de {fmt(shunt['resistencia_ohm'])} ohm em série com o transmissor):")
    print(f"  V = R x I = {fmt(shunt['resistencia_ohm'])} x {fmt(shunt['corrente_a'])} = "
          f"{fmt(shunt['tensao_v'], 2)} V; o sensor mede essa queda e calcula I = V / R")

    secao("Dispositivos de entrada, saída e interfaces do SCIC")
    for papel, dispositivo, uso, interface in HW.DISPOSITIVOS_ES:
        print(f"{papel:<10}{dispositivo}")
        print(f"{'':<10}{uso} | {interface}")
    informar_graficos(sessao, [HW.grafico_consumo(df)])


# ==========================================================================
# 9 - gerenciamento inteligente
# ==========================================================================
def opcao_gestao(sessao: Sessao) -> None:
    titulo("9 - Gerenciamento inteligente da comunicação")
    df = sessao.preparar()
    anomalias = G.detectar_anomalias(df)
    contagem = (anomalias.groupby("modulo")["anomalia"].sum().astype(int)
                .sort_values(ascending=False, kind="mergesort"))   # empate: ordem alfabética
    secao("Monitoramento contínuo: anomalias de latência")
    escrever(f"Regra: latência acima da média dos {G.JANELA_MOVEL} sóis anteriores do próprio módulo "
             f"+ {fmt(G.FATOR_DESVIO, 0)} desvios-padrão (sem olhar o futuro).")
    escrever(f"{int(contagem.sum())} anomalias nos enlaces internos. Por módulo: "
             + ", ".join(f"{m} {n}" for m, n in contagem.items() if n > 0) + ".")

    por_sol = G.eventos_sistemicos(df)
    eventos = por_sol[por_sol["evento_sistemico"]]
    secao("Evento sistêmico x falha local")
    for sol, linha in eventos.iterrows():
        print(f"Sol {sol}: {int(linha['modulos_em_alerta'])} módulos em alerta, qualidade média "
              f"{fmt(linha['qualidade_media'])}% e perda média {fmt(linha['perda_media'], 2)}%")
    if len(eventos):
        escrever("Causa comum (tempestade de poeira): a automação recomenda o enlace reserva e adia "
                 "transmissões não críticas, em vez de abrir uma ordem de serviço por módulo.")
    else:
        print("Nenhum sol com metade da colônia em alerta.")

    tendencias = G.tendencia_degradacao(df)
    secao("Manutenção preditiva: tendência da qualidade do sinal")
    exibicao = tendencias.head(4).copy()
    exibicao["sol_cruza_70"] = exibicao["sol_cruza_70"].where(exibicao["sol_cruza_70"] <= 100)
    exibicao.index.name = "módulo"
    mostrar_tabela(exibicao.rename(columns={
        "inclinacao_por_sol": "p.p. por sol", "qualidade_tendencia_atual": "tendência hoje %",
        "sol_cruza_70": "cruza 70% no sol", "recomendacao": "recomendação"}),
        {"p.p. por sol": 2, "tendência hoje %": 1, "cruza 70% no sol": 1})
    print("('-' = não cruza 70% nos próximos 100 sóis; p.p. = pontos percentuais)")
    ant = G.previsao_antecipada(df)
    if ant["sol_previsto"] == ant["sol_previsto"]:          # não é NaN
        escrever(f"Antecipação: com os dados só até o sol {ant['ate_sol']}, a reta {do_modulo(ant['modulo'])} "
                 f"já previa o cruzamento de 70% no sol {fmt(ant['sol_previsto'], 0)}: "
                 f"{fmt(ant['antecedencia_sois'], 0)} sóis para agendar a manutenção antes da falha.")

    red = G.simular_redundancia(df)
    secao("Redundância de enlaces")
    print(f"Enlace principal fora do ar (alerta ou manutenção) em {fmt(red['p_falha'] * 100)}% dos sóis.")
    print(f"1 enlace:  disponibilidade 1 - p  = {fmt(red['disponibilidade_1_enlace'] * 100)}% "
          f"({fmt(red['horas_fora_1_enlace'], 0)} h fora em {red['sois_analisados']} sóis)")
    print(f"2 enlaces: disponibilidade 1 - p² = {fmt(red['disponibilidade_2_enlaces'] * 100)}% "
          f"({fmt(red['horas_fora_2_enlaces'], 0)} h fora), com falhas independentes")

    ef = G.eficiencia_energetica(df)
    secao("Eficiência: energia gasta com retransmissões")
    print(f"Consumo dos transceptores em 30 sóis: {fmt(ef['consumo_total_kwh'])} kWh")
    print(f"Retransmissões por perda de pacotes: {fmt(ef['retransmissao_kwh'], 2)} kWh "
          f"({fmt(ef['retransmissao_pct'], 2)}% da energia)")
    escrever(f"Levando a perda a no máximo {fmt(G.PERDA_ALVO_PCT)}%, a colônia economizaria "
             f"{fmt(ef['economia_possivel_kwh'], 2)} kWh a cada 30 sóis. Maior gasto: "
             f"{ef['por_modulo_kwh'].index[0]} ({fmt(ef['por_modulo_kwh'].iloc[0], 2)} kWh).")
    informar_graficos(sessao, [G.grafico_tendencia(df, tendencias), G.grafico_alertas_por_sol(por_sol),
                               G.grafico_anomalias(anomalias)])


# ==========================================================================
# 10 - análise final
# ==========================================================================
def montar_analise_final(sessao: Sessao) -> list[str]:
    """Junta os principais números de todas as opções num resumo para apoiar decisões."""
    df = sessao.preparar()
    r = sessao.relatorio_limpeza
    _, resumo = D.calcular_indicadores(df)
    erros_mod = E.resumo_erros_por_modulo(df)
    escala = E.comparar_escalas(df)
    euler = E.analise_euler(df)
    resultado = sessao.modelo()
    final = resultado["tabela"].loc[resultado["nome_final"]]
    engenharia = resultado["tabela"].loc["Estimativa de engenharia (CSV)"]
    heap = H.montar_fila_alertas(df)
    score, alerta = heap.topo()
    bench_heap = sessao.benchmark_heap().iloc[-1]
    bench_trie = sessao.benchmark_trie().iloc[-1]
    tendencias = G.tendencia_degradacao(df)
    pior = tendencias.iloc[0]
    antecipada = G.previsao_antecipada(df, pior.name)
    eventos = G.eventos_sistemicos(df)
    sois_evento = [str(s) for s in eventos.index[eventos["evento_sistemico"]]]
    red = G.simular_redundancia(df)
    ef = G.eficiencia_energetica(df)
    criticos = erros_mod[erros_mod["faixa_media"] == "crítico"]
    criticos_vida = [m for m, p in criticos["prioridade"].items() if p >= 5]
    criticos_padrao = [m for m, p in criticos["prioridade"].items() if p < 5]
    partes_criticos = []
    if criticos_padrao:
        partes_criticos.append(f"{', '.join(criticos_padrao)} (acima de {fmt(E.LIMITES_PADRAO[1], 0)}%)")
    if criticos_vida:
        partes_criticos.append(f"{', '.join(criticos_vida)} (acima de {fmt(E.LIMITES_VIDA_HUMANA[1], 0)}%, "
                               "limite de quem mantém vidas)")
    reducao = (1 - final["MAE"] / engenharia["MAE"]) * 100
    ultimo_sol = int(df["ciclo"].max())
    verbo = "cruzou" if pior["sol_cruza_70"] <= ultimo_sol else "deve cruzar"

    if pior["recomendacao"] == "manutenção corretiva já":
        acao_sinal = (f"Fazer já a manutenção corretiva no enlace {do_modulo(pior.name)}: a tendência "
                      "da qualidade do sinal está abaixo de 70%.")
    elif pior["recomendacao"] == "agendar manutenção preditiva":
        acao_sinal = (f"Agendar a manutenção preditiva do enlace {do_modulo(pior.name)} antes do sol "
                      f"{fmt(pior['sol_cruza_70'], 0)}.")
    else:
        acao_sinal = "Manter o monitoramento: nenhum módulo perde sinal de forma preocupante."

    return [
        "ANÁLISE FINAL - SCIC Aurora Siger",
        "",
        "Dados",
        f"- {r['linhas_finais']} registros válidos de {resumo['modulos']} módulos em {resumo['sois']} sóis "
        f"({r['linhas_lidas']} lidos; {r['duplicadas_removidas']} duplicado e "
        f"{r['sem_latencia_removidas']} sem latência removidos; "
        f"{sum(r['valores_imputados'].values())} vazios completados).",
        f"- {fmt(resumo['pct_alerta'])}% dos registros em alerta; disponibilidade média de "
        f"{fmt(resumo['disponibilidade_media_pct'])}%. Menor: {resumo['pior_disponibilidade'][0]} "
        f"({fmt(resumo['pior_disponibilidade'][1])}%).",
        f"- Latência interna média de {fmt(resumo['latencia_media_interna_ms'])} ms; enlace Terra-Marte "
        f"de {fmt(resumo['latencia_min_terra_min'], 2)} a {fmt(resumo['latencia_max_terra_min'], 2)} min.",
        "",
        "Erros numéricos",
        f"- A estimativa de engenharia erra {fmt(escala['interno_erro_rel_pct'])}% em média nos enlaces "
        f"internos e subestima a latência em {fmt(escala['vies_interno_ms'])} ms.",
        f"- Erro médio na faixa crítica: {'; '.join(partes_criticos) if partes_criticos else 'nenhum módulo'}.",
        f"- Terra-Marte: erro absoluto de {fmt(escala['terra_erro_abs_ms'], 0)} ms, mas relativo de só "
        f"{fmt(escala['terra_erro_rel_pct'], 2)}%.",
        f"- Euler com h = 0,1 erra {fmt(euler['tabela']['erro_rel_pct'].iloc[-1], 3)}% contra a solução "
        f"exata; Newton acha o cruzamento de 70% em {euler['iteracoes_newton']} iterações.",
        "",
        "Modelo de previsão",
        f"- {M.nome_curto(resultado['nome_final'])}: MAE {fmt(final['MAE'])} ms, RMSE {fmt(final['RMSE'])} ms "
        f"e R² {fmt(final['R2'], 2)} no teste.",
        f"- Erro médio {reducao:.0f}% menor que o da estimativa de engenharia (MAE "
        f"{fmt(engenharia['MAE'])} ms). Sem os {resultado['n_picos_teste']} picos de retransmissão, "
        f"R² = {fmt(resultado['metricas_sem_picos']['R2'], 2)}.",
        "",
        "Alertas e busca",
        f"- {len(heap)} alertas abertos nos últimos {H.JANELA_SOIS} sóis. Mais urgente: {alerta['modulo']}, "
        f"sol {alerta['ciclo']}, {alerta['mensagem'].lower()} (score {fmt(score, 2)}).",
        f"- Heap {bench_heap['vantagem_x']:.0f} vezes mais rápido que a lista com "
        f"{fmt(bench_heap['n_alertas'], 0)} alertas; trie {bench_trie['vantagem_x']:.0f} vezes mais rápida "
        f"que a busca linear com {fmt(bench_trie['n_codigos'], 0)} códigos.",
        "",
        "Gestão inteligente",
        f"- {pior.name}: o sinal perde {fmt(abs(pior['inclinacao_por_sol']), 2)} p.p. por sol e a tendência "
        f"{verbo} 70% no sol {fmt(pior['sol_cruza_70'], 0)} (Newton: sol {fmt(euler['sol_cruzamento'])}). "
        f"Com dados só até o sol {antecipada['ate_sol']}, já era possível prever o sol "
        f"{fmt(antecipada['sol_previsto'], 0)}.",
        f"- Evento sistêmico nos sóis {' e '.join(sois_evento) if sois_evento else '-'} (tempestade de "
        "poeira): resposta coletiva, não chamados isolados.",
        f"- Enlace reserva: disponibilidade de {fmt(red['disponibilidade_1_enlace'] * 100)}% para "
        f"{fmt(red['disponibilidade_2_enlaces'] * 100)}% ({fmt(red['horas_fora_1_enlace'], 0)} h para "
        f"{fmt(red['horas_fora_2_enlaces'], 0)} h fora em 30 sóis).",
        f"- Retransmissões gastam {fmt(ef['retransmissao_kwh'], 2)} kWh a cada 30 sóis; "
        f"{fmt(ef['economia_possivel_kwh'], 2)} kWh podem ser economizados.",
        "",
        "Recomendações",
        f"1. {acao_sinal}",
        f"2. Atender primeiro o alerta {do_modulo(alerta['modulo'])} no sol {alerta['ciclo']} "
        f"({alerta['mensagem'].lower()}), o topo do heap.",
        f"3. Trocar a estimativa fixa de engenharia pelo modelo, que considera a carga real "
        f"(erro médio {reducao:.0f}% menor).",
        "4. Instalar o enlace reserva e reduzir a perda de pacotes, que gasta energia com retransmissões.",
        "5. Manter a validação humana antes de qualquer ação automática.",
    ]


def opcao_analise_final(sessao: Sessao) -> None:
    titulo("10 - Análise final dos resultados")
    if sessao.bench_heap is None or sessao.bench_trie is None:
        print("Reunindo os resultados (inclui a medição de tempo do heap e da trie)...")
    linhas = montar_analise_final(sessao)
    for linha in linhas[2:]:
        if linha.startswith(("- ", "1.", "2.", "3.", "4.", "5.")):
            escrever(linha, recuo="  " if linha.startswith("- ") else "   ")
        else:
            print(linha)
    destino = GR.PASTA_GRAFICOS / "resumo_analise_final.txt"
    GR.PASTA_GRAFICOS.mkdir(exist_ok=True)
    destino.write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print(f"\nResumo salvo em graficos_ou_imagens/{destino.name}")


# ==========================================================================
# menu
# ==========================================================================
ACOES = {1: opcao_carregar, 2: opcao_consultar, 3: opcao_indicadores_erros, 4: opcao_previsao,
         5: opcao_avaliacao, 6: opcao_heap, 7: opcao_trie, 8: opcao_hardware, 9: opcao_gestao,
         10: opcao_analise_final}


def mostrar_menu() -> None:
    titulo("SCIC - Sistema de Comunicação Interplanetária da Colônia Aurora Siger")
    for numero, descricao in OPCOES:
        print(f"  {numero:>2}  {descricao}")
    print("Digite o número ou o começo de um comando (ex.: 'pri' = priorizar alertas).")


def resolver_escolha(sessao: Sessao, texto: str) -> int | None:
    """Número da opção, ou comando por prefixo resolvido pela trie de comandos."""
    texto = texto.strip()
    if not texto:
        return None
    if texto.isdigit():
        numero = int(texto)
        return numero if numero in ACOES or numero == 0 else None
    achados = sessao.trie_comandos.buscar_prefixo(texto)
    opcoes = sorted({refs[0] for _, refs in achados})
    if len(opcoes) == 1:
        nome = dict(OPCOES)[opcoes[0]]
        print(f"  '{texto}' -> opção {opcoes[0]} ({nome})")
        return opcoes[0]
    if achados:
        print("  Comando ambíguo: " + ", ".join(f"{t} ({r[0]})" for t, r in achados))
    return None


def executar_demo() -> None:
    """Roda as 10 opções com os valores padrão: serve de exemplo de execução e de teste."""
    inicio = time.perf_counter()
    sessao = Sessao(demo=True)
    for numero in range(1, 11):
        ACOES[numero](sessao)
    print(f"\nDemonstração concluída em {fmt(time.perf_counter() - inicio)} s. "
          "Rode sem --demo para usar o menu interativo.")


def main() -> None:
    if "--demo" in sys.argv:
        executar_demo()
        return
    sessao = Sessao()
    try:
        sessao.preparar()
    except (FileNotFoundError, ValueError) as erro:
        print(f"Erro ao carregar a base: {erro}")
        return
    while True:
        mostrar_menu()
        try:
            escolha = resolver_escolha(sessao, input("Opção: "))
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o SCIC.")
            return
        if escolha is None:
            print("  Opção inválida. Digite um número de 0 a 10 ou o começo de um comando.")
            continue
        if escolha == 0:
            print("Encerrando o SCIC. Até a próxima transmissão!")
            return
        try:
            ACOES[escolha](sessao)
        except (KeyboardInterrupt, EOFError):
            print("\nOperação cancelada. Voltando ao menu.")
        try:
            input("\nEnter para voltar ao menu...")
        except (KeyboardInterrupt, EOFError):
            print("\nEncerrando o SCIC.")
            return


if __name__ == "__main__":
    main()
