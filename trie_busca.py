"""
trie_busca.py
=============
Busca por prefixo com uma trie (árvore de prefixos) implementada do zero.

Cada caminho da raiz até um nó soletra um prefixo; palavras que começam igual
compartilham o mesmo caminho. Buscar um prefixo custa O(m), em que m é o
tamanho do prefixo, independentemente de quantos termos estão guardados.

Adaptações em relação à trie da aula (em C, com vetor de 26 letras):
* os filhos ficam num dicionário, porque os termos do SCIC têm acentos,
  espaços, dígitos e "0x" (códigos hexadecimais);
* o texto é normalizado (minúsculas, sem acento), então "comunicacao"
  encontra "Comunicação";
* além da busca exata da aula, há a busca por prefixo (autocomplete): desce
  até o nó do prefixo e percorre a subárvore coletando os termos completos.
"""
from __future__ import annotations

import gc
import random
import time
from pathlib import Path

import pandas as pd

import graficos as g
from dados import normalizar_texto

# comandos do menu: o usuário pode digitar "pri" em vez de "6"
COMANDOS_MENU = {
    "carregar": 1, "cadastrar": 1, "consultar": 2, "indicadores": 3, "erros": 3,
    "prever": 4, "previsao": 4, "avaliar": 5, "metricas": 5, "priorizar": 6, "alertas": 6,
    "heap": 6, "buscar": 7, "trie": 7, "hardware": 8, "bases": 8, "eletricidade": 8,
    "gestao": 9, "monitoramento": 9, "analise": 10, "final": 10, "sair": 0,
}


class NoTrie:
    """Nó da trie: filhos por caractere, marca de fim de termo e referências."""
    __slots__ = ("filhos", "fim", "referencias", "original")

    def __init__(self) -> None:
        self.filhos: dict[str, NoTrie] = {}
        self.fim = False
        self.referencias: list = []
        self.original = ""


class Trie:
    """Trie com inserção, busca exata e busca por prefixo."""

    def __init__(self) -> None:
        self.raiz = NoTrie()
        self.total_termos = 0

    def inserir(self, termo: str, referencia=None) -> None:
        """Percorre o termo letra a letra, criando só os nós que ainda não existem."""
        no = self.raiz
        for caractere in normalizar_texto(termo):
            if caractere not in no.filhos:
                no.filhos[caractere] = NoTrie()
            no = no.filhos[caractere]
        if not no.fim:
            no.fim = True
            no.original = termo
            self.total_termos += 1
        if referencia is not None and referencia not in no.referencias:
            no.referencias.append(referencia)

    def _descer(self, texto: str) -> NoTrie | None:
        no = self.raiz
        for caractere in normalizar_texto(texto):
            no = no.filhos.get(caractere)
            if no is None:
                return None
        return no

    def buscar(self, termo: str) -> list | None:
        """Busca exata: devolve as referências do termo ou None se ele não existe."""
        no = self._descer(termo)
        return no.referencias if no is not None and no.fim else None

    def buscar_prefixo(self, prefixo: str, limite: int | None = None) -> list[tuple[str, list]]:
        """Autocomplete: todos os termos que começam com o prefixo, em ordem alfabética."""
        no = self._descer(prefixo)
        if no is None:
            return []
        encontrados = []
        pilha = [no]
        while pilha:                       # busca em profundidade na subárvore
            atual = pilha.pop()
            if atual.fim:
                encontrados.append((atual.original, atual.referencias))
            pilha.extend(atual.filhos.values())
        encontrados.sort(key=lambda par: normalizar_texto(par[0]))
        return encontrados[:limite] if limite else encontrados

    def contar_nos(self) -> int:
        total, pilha = 0, [self.raiz]
        while pilha:
            atual = pilha.pop()
            total += 1
            pilha.extend(atual.filhos.values())
        return total


def construir_indice(df: pd.DataFrame) -> dict[str, Trie]:
    """Uma trie por tipo de termo: módulos, códigos de sensor, palavras de alerta, comandos."""
    indice = {"módulos": Trie(), "códigos de sensor": Trie(),
              "palavras de alerta": Trie(), "comandos do menu": Trie()}
    for modulo, codigo in df[["modulo", "codigo_sensor"]].drop_duplicates().itertuples(index=False):
        indice["módulos"].inserir(modulo, modulo)
        for palavra in modulo.split():            # "medico" também acha "Centro Médico"
            if len(palavra) >= 3:
                indice["módulos"].inserir(palavra, modulo)
        indice["códigos de sensor"].inserir(codigo, modulo)
    for id_registro, mensagem in df[["id_registro", "mensagem_alerta"]].itertuples(index=False):
        for palavra in str(mensagem).replace("-", " ").split():
            if len(palavra) >= 4:
                indice["palavras de alerta"].inserir(palavra.lower(), int(id_registro))
    for comando, opcao in COMANDOS_MENU.items():
        indice["comandos do menu"].inserir(comando, opcao)
    return indice


def buscar_em_tudo(indice: dict[str, Trie], prefixo: str) -> dict[str, list[tuple[str, list]]]:
    """Busca o prefixo em todas as tries e agrupa os resultados por categoria."""
    resultados = {}
    for categoria, trie in indice.items():
        achados = trie.buscar_prefixo(prefixo)
        if categoria == "módulos":   # palavras soltas apontam para o nome completo
            nomes = sorted({ref for _, refs in achados for ref in refs}, key=normalizar_texto)
            achados = [(nome, [nome]) for nome in nomes]
        if achados:
            resultados[categoria] = achados
    return resultados


# --------------------------------------------------------------------------
# benchmark: trie x busca linear
# --------------------------------------------------------------------------
def benchmark_trie_vs_linear(tamanhos=(5_000, 10_000, 20_000, 40_000), consultas: int = 150,
                             repeticoes: int = 3, seed: int = 42) -> pd.DataFrame:
    """Tempo médio por consulta de prefixo em bases de códigos cada vez maiores.

    Busca linear: compara o prefixo com todos os códigos -> O(n * m).
    Trie: desce m níveis e coleta só o que começa com o prefixo -> O(m + k).
    Cada medição é repetida e vale o menor tempo, com o coletor de lixo desligado.
    """
    gerador = random.Random(seed)
    linhas = []
    for n in tamanhos:
        unicos: set[str] = set()
        while len(unicos) < n:                      # exatamente n códigos diferentes
            unicos.add(f"0x{gerador.getrandbits(32):08X}")
        codigos = sorted(unicos)
        trie = Trie()
        for codigo in codigos:
            trie.inserir(codigo)
        prefixos = [gerador.choice(codigos)[:7] for _ in range(consultas)]
        prefixos_norm = [normalizar_texto(p) for p in prefixos]
        codigos_norm = [normalizar_texto(c) for c in codigos]

        tempos_linear, tempos_trie = [], []
        gc_ligado = gc.isenabled()
        gc.disable()
        try:
            for _ in range(repeticoes):
                inicio = time.perf_counter()
                for p in prefixos_norm:
                    [c for c in codigos_norm if c.startswith(p)]
                tempos_linear.append((time.perf_counter() - inicio) / consultas)

                inicio = time.perf_counter()
                for p in prefixos:
                    trie.buscar_prefixo(p)
                tempos_trie.append((time.perf_counter() - inicio) / consultas)
        finally:
            if gc_ligado:
                gc.enable()
        linhas.append({"n_codigos": n, "linear_ms": min(tempos_linear) * 1000,
                       "trie_ms": min(tempos_trie) * 1000})
    tabela = pd.DataFrame(linhas)
    tabela["vantagem_x"] = tabela["linear_ms"] / tabela["trie_ms"]
    return tabela


def grafico_benchmark(tabela: pd.DataFrame) -> Path:
    """Tempo médio por consulta: a trie fica estável, a busca linear cresce com n."""
    fig, ax = g.nova_figura(8, 4.4)
    ax.plot(tabela["n_codigos"], tabela["linear_ms"], "o-", color=g.LARANJA, markersize=7,
            markeredgecolor="white", label="Busca linear  O(n·m)")
    ax.plot(tabela["n_codigos"], tabela["trie_ms"], "o-", color=g.AZUL, markersize=7,
            markeredgecolor="white", label="Trie  O(m + k)")
    ultimo = tabela.iloc[-1]
    ax.text(ultimo["n_codigos"], ultimo["linear_ms"], f"  {g.fmt(ultimo['linear_ms'], 2)} ms",
            va="center", fontsize=8.5, color=g.TINTA_2)
    ax.text(ultimo["n_codigos"], ultimo["trie_ms"], f"  {g.fmt(ultimo['trie_ms'], 3)} ms",
            va="center", fontsize=8.5, color=g.TINTA_2)
    ax.set_xlim(0, ultimo["n_codigos"] * 1.18)
    ax.set_xlabel("Códigos de sensor guardados")
    ax.set_ylabel("Tempo médio por consulta (ms)")
    ax.set_title(f"Com {g.fmt(ultimo['n_codigos'], 0)} códigos, a trie responde "
                 f"{ultimo['vantagem_x']:.0f} vezes mais rápido")
    g.subtitulo(ax, "Consultas por prefixo de 7 caracteres, como 0x1A2B3")
    ax.legend(loc="upper left")
    return g.salvar_figura(fig, "trie_benchmark.png")
