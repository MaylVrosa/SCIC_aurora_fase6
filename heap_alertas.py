"""
heap_alertas.py
===============
Priorização dos alertas da colônia com um max-heap implementado do zero.

O heap é uma árvore binária quase completa guardada numa lista:
    pai de i = (i - 1) // 2      filhos de i = 2i + 1 e 2i + 2
Regra do max-heap: todo pai tem prioridade maior ou igual à dos filhos, então o
alerta mais urgente está sempre na posição 0.

* inserir  -> coloca no fim e faz heapify-up (sobe trocando com o pai)   O(log n)
* extrair  -> tira a raiz, põe o último no lugar e faz heapify-down      O(log n)
* topo     -> lê a posição 0                                             O(1)

É a mesma lógica do código em C da aula (heapifyUp, heapifyDown,
construirMaxHeap e heapSort), escrita em Python.
"""
from __future__ import annotations

import gc
import heapq
import json
import random
import time
from pathlib import Path

import pandas as pd

import graficos as g

JANELA_SOIS = 10   # alertas registrados nos últimos 10 sóis estão "abertos"


class MaxHeapAlertas:
    """Fila de prioridade de alertas: cada item é (prioridade, ordem de chegada, alerta)."""

    def __init__(self) -> None:
        self._dados: list[tuple[float, int, dict]] = []
        self._chegadas = 0

    def __len__(self) -> int:
        return len(self._dados)

    # -- comparação: maior prioridade vence; empate -> quem chegou antes ---------
    def _mais_urgente(self, i: int, j: int) -> bool:
        pi, oi, _ = self._dados[i]
        pj, oj, _ = self._dados[j]
        return (pi, -oi) > (pj, -oj)

    def _trocar(self, i: int, j: int) -> None:
        self._dados[i], self._dados[j] = self._dados[j], self._dados[i]

    # -- operações principais ------------------------------------------------
    def inserir(self, prioridade: float, alerta: dict) -> None:
        """Insere no fim da lista e sobe o item até a posição certa (heapify-up)."""
        self._chegadas += 1
        self._dados.append((prioridade, self._chegadas, alerta))
        self._heapify_up(len(self._dados) - 1)

    def _heapify_up(self, i: int) -> None:
        while i > 0:
            pai = (i - 1) // 2
            if self._mais_urgente(i, pai):
                self._trocar(i, pai)
                i = pai
            else:
                break

    def item_topo(self) -> tuple[float, int, dict] | None:
        """Tupla completa guardada na raiz: (prioridade, ordem de chegada, alerta)."""
        return self._dados[0] if self._dados else None

    def topo(self) -> tuple[float, dict] | None:
        """Mostra o alerta mais urgente sem retirá-lo."""
        if not self._dados:
            return None
        prioridade, _, alerta = self._dados[0]
        return prioridade, alerta

    def extrair_mais_urgente(self) -> tuple[float, dict] | None:
        """Retira a raiz, coloca o último item no lugar e desce (heapify-down)."""
        if not self._dados:
            return None
        prioridade, _, alerta = self._dados[0]
        ultimo = self._dados.pop()
        if self._dados:
            self._dados[0] = ultimo
            self._heapify_down(0)
        return prioridade, alerta

    def _heapify_down(self, i: int) -> None:
        n = len(self._dados)
        while True:
            maior, esquerda, direita = i, 2 * i + 1, 2 * i + 2
            if esquerda < n and self._mais_urgente(esquerda, maior):
                maior = esquerda
            if direita < n and self._mais_urgente(direita, maior):
                maior = direita
            if maior == i:
                return
            self._trocar(i, maior)
            i = maior

    @classmethod
    def construir(cls, itens: list[tuple[float, dict]]) -> "MaxHeapAlertas":
        """Constrói o heap de uma vez em O(n): heapify-down do último pai até a raiz."""
        heap = cls()
        for prioridade, alerta in itens:
            heap._chegadas += 1
            heap._dados.append((prioridade, heap._chegadas, alerta))
        for i in range(len(heap._dados) // 2 - 1, -1, -1):
            heap._heapify_down(i)
        return heap

    def vetor(self) -> list[float]:
        """As prioridades na ordem em que estão guardadas na lista (para explicar a árvore)."""
        return [item[0] for item in self._dados]

    def valido(self) -> bool:
        """Confere a propriedade do max-heap em todos os pares pai-filho."""
        return all(not self._mais_urgente(i, (i - 1) // 2) for i in range(1, len(self._dados)))


def heap_sort(itens: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    """Ranking completo: constrói o heap e extrai a raiz n vezes. O(n log n)."""
    heap = MaxHeapAlertas.construir(itens)
    return [heap.extrair_mais_urgente() for _ in range(len(heap))]


# --------------------------------------------------------------------------
# critério de prioridade
# --------------------------------------------------------------------------
def calcular_score(registro: pd.Series, ultimo_sol: int) -> tuple[float, dict]:
    """Prioridade transparente de um alerta, somando cinco critérios.

    3 x prioridade do módulo (5 = vida humana)      -> 3 a 15 pontos
    erro relativo da previsão / 10 (teto de 50%)     -> 0 a 5 pontos
    queda de tensão no transmissor                   -> +2 pontos
    qualidade < 70% e perda de pacotes > 4%          -> +1 ponto cada
    0,2 por sol de espera (tempo desde o registro)   -> evita que um alerta fique esquecido
    """
    erro = registro.get("erro_rel_pct", 0.0)
    erro = 0.0 if pd.isna(erro) else float(erro)
    partes = {
        "modulo": 3.0 * int(registro["prioridade_base"]),
        "erro_previsao": min(erro, 50.0) / 10.0,
        "tensao": 2.0 if float(registro["tensao_v"]) < 25.2 else 0.0,
        "sinal": (1.0 if float(registro["qualidade_sinal_pct"]) < 70 else 0.0)
        + (1.0 if float(registro["perda_pacotes_pct"]) > 4 else 0.0),
        "espera": 0.2 * (ultimo_sol - int(registro["ciclo"])),
    }
    return round(sum(partes.values()), 2), partes


def alertas_abertos(df: pd.DataFrame, janela: int = JANELA_SOIS) -> pd.DataFrame:
    """Registros em alerta dentro da janela de sóis mais recentes."""
    ultimo = int(df["ciclo"].max())
    return df[(df["status"] == "alerta") & (df["ciclo"] > ultimo - janela)]


def montar_fila_alertas(df: pd.DataFrame, janela: int = JANELA_SOIS) -> MaxHeapAlertas:
    """Cria o heap com os alertas abertos, um a um (heapify-up a cada inserção)."""
    ultimo = int(df["ciclo"].max())
    heap = MaxHeapAlertas()
    for _, registro in alertas_abertos(df, janela).iterrows():
        score, partes = calcular_score(registro, ultimo)
        heap.inserir(score, {
            "id_registro": int(registro["id_registro"]),
            "ciclo": int(registro["ciclo"]),
            "modulo": registro["modulo"],
            "codigo_sensor": registro["codigo_sensor"],
            "mensagem": registro["mensagem_alerta"],
            "erro_rel_pct": round(float(registro.get("erro_rel_pct", 0) or 0), 1),
            "tensao_v": float(registro["tensao_v"]),
            "componentes": partes,
        })
    return heap


def conferir_com_heapq(df: pd.DataFrame, k: int = 5, janela: int = JANELA_SOIS) -> bool:
    """Compara o top-k do nosso heap com o heapq da biblioteca padrão do Python."""
    ultimo = int(df["ciclo"].max())
    nosso = montar_fila_alertas(df, janela)
    top_nosso = [nosso.extrair_mais_urgente()[1]["id_registro"] for _ in range(min(k, len(nosso)))]
    fila = []
    for ordem, (_, registro) in enumerate(alertas_abertos(df, janela).iterrows()):
        score, _ = calcular_score(registro, ultimo)
        heapq.heappush(fila, (-score, ordem, int(registro["id_registro"])))  # heapq é min-heap
    top_heapq = [heapq.heappop(fila)[2] for _ in range(min(k, len(fila)))]
    return top_nosso == top_heapq


def payload_webhook(score: float, alerta: dict) -> str:
    """JSON que um fluxo no n8n receberia por webhook para avisar a equipe (simulado)."""
    corpo = {
        "evento": "alerta_critico_scic",
        "prioridade": score,
        "modulo": alerta["modulo"],
        "codigo_sensor": alerta["codigo_sensor"],
        "sol": alerta["ciclo"],
        "mensagem": alerta["mensagem"],
        "acao_sugerida": "validar com a equipe humana antes de qualquer ação automática",
    }
    return json.dumps(corpo, ensure_ascii=False, indent=2)


# --------------------------------------------------------------------------
# benchmark: heap x lista simples
# --------------------------------------------------------------------------
def benchmark_heap_vs_lista(tamanhos=(1000, 2000, 4000, 8000), repeticoes: int = 3,
                            seed: int = 42) -> pd.DataFrame:
    """Simula o monitoramento: a cada alerta novo, o sistema consulta o mais urgente.

    Lista simples: inserir é O(1), mas achar o maior com max() é O(n) -> total O(n²).
    Heap: inserir é O(log n) e o maior está sempre no topo -> total O(n log n).
    Cada medição é repetida e vale o menor tempo, com o coletor de lixo desligado,
    como faz o módulo timeit: assim uma pausa do sistema não distorce o resultado.
    """
    gerador = random.Random(seed)
    linhas = []
    for n in tamanhos:
        prioridades = [gerador.uniform(0, 25) for _ in range(n)]
        tempos_lista, tempos_heap = [], []
        gc_ligado = gc.isenabled()
        gc.disable()
        try:
            for _ in range(repeticoes):
                inicio = time.perf_counter()
                lista = []
                for p in prioridades:
                    lista.append(p)
                    max(lista)
                tempos_lista.append(time.perf_counter() - inicio)

                inicio = time.perf_counter()
                heap = MaxHeapAlertas()
                for p in prioridades:
                    heap.inserir(p, {})
                    heap.topo()
                tempos_heap.append(time.perf_counter() - inicio)
        finally:
            if gc_ligado:
                gc.enable()
        linhas.append({"n_alertas": n, "lista_ms": min(tempos_lista) * 1000,
                       "heap_ms": min(tempos_heap) * 1000})
    tabela = pd.DataFrame(linhas)
    tabela["vantagem_x"] = tabela["lista_ms"] / tabela["heap_ms"]
    return tabela


def grafico_benchmark(tabela: pd.DataFrame) -> Path:
    """Tempo total de inserir n alertas consultando o mais urgente a cada chegada."""
    fig, ax = g.nova_figura(8, 4.4)
    ax.plot(tabela["n_alertas"], tabela["lista_ms"], "o-", color=g.LARANJA, markersize=7,
            markeredgecolor="white", label="Lista simples + max()  O(n²)")
    ax.plot(tabela["n_alertas"], tabela["heap_ms"], "o-", color=g.AZUL, markersize=7,
            markeredgecolor="white", label="Max-heap  O(n log n)")
    ultimo = tabela.iloc[-1]
    ax.text(ultimo["n_alertas"], ultimo["lista_ms"], f"  {ultimo['lista_ms']:.0f} ms",
            va="center", fontsize=8.5, color=g.TINTA_2)
    ax.text(ultimo["n_alertas"], ultimo["heap_ms"], f"  {ultimo['heap_ms']:.0f} ms",
            va="center", fontsize=8.5, color=g.TINTA_2)
    ax.set_xlim(0, ultimo["n_alertas"] * 1.15)
    ax.set_xlabel("Alertas recebidos")
    ax.set_ylabel("Tempo total (ms)")
    ax.set_title(f"Com {g.fmt(ultimo['n_alertas'], 0)} alertas, o heap é "
                 f"{ultimo['vantagem_x']:.0f} vezes mais rápido")
    g.subtitulo(ax, "A cada alerta novo, o sistema consulta qual é o mais urgente")
    ax.legend(loc="upper left")
    return g.salvar_figura(fig, "heap_benchmark.png")
