"""
graficos.py
===========
Estilo visual único para todos os gráficos do SCIC, funções para salvar e abrir
as imagens geradas na pasta graficos_ou_imagens/ e a formatação de números no
padrão brasileiro (vírgula decimal), usada nos gráficos e no terminal.

Os gráficos usam o backend "Agg" do Matplotlib: são salvos como PNG sem abrir
janelas, então o menu nunca trava esperando uma janela ser fechada. Quando o
usuário quer ver a imagem, ela é aberta no visualizador padrão do sistema.
"""
from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva sem abrir janela (funciona em qualquer terminal)
import matplotlib.pyplot as plt  # noqa: E402  (precisa vir depois do use)
from matplotlib.ticker import ScalarFormatter  # noqa: E402

PASTA_GRAFICOS = Path(__file__).resolve().parent / "graficos_ou_imagens"

# Paleta: tinta (textos), cinzas de apoio e cores de série em ordem fixa
TINTA = "#0b0b0b"
TINTA_2 = "#52514e"
TINTA_FRACA = "#898781"
GRADE = "#e1e0d9"
EIXO = "#c3c2b7"
CINZA_CONTEXTO = "#c9c7bd"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
AZUL, LARANJA, VERDE_AGUA = SERIES[0], SERIES[1], SERIES[2]

# Cores de status: usadas só quando a cor significa bom / atenção / crítico
STATUS_COR = {"aceitável": "#0ca30c", "atenção": "#fab219", "crítico": "#d03b3b"}


def fmt(valor: float, casas: int = 1) -> str:
    """Número no padrão brasileiro: 1.234,5 (ponto no milhar, vírgula decimal)."""
    if valor is None or valor != valor:          # None ou NaN (NaN é diferente de si mesmo)
        return "-"
    texto = f"{valor:,.{casas}f}"
    return texto.replace(",", "X").replace(".", ",").replace("X", ".")


class FormatadorBR(ScalarFormatter):
    """Marcas dos eixos no padrão brasileiro: 0,5 em vez de 0.5 e 80.000 em vez de 80000."""

    def __call__(self, x, pos=None):
        texto = super().__call__(x, pos)
        digitos = texto.lstrip("-\u2212")
        if digitos.isdigit() and len(digitos) > 4:
            return texto[: len(texto) - len(digitos)] + f"{int(digitos):,}".replace(",", ".")
        return texto.replace(".", ",")


def aplicar_estilo() -> None:
    """Configura o Matplotlib com eixos discretos, grade fina e textos em tinta."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": EIXO,
        "axes.labelcolor": TINTA_2,
        "axes.titlecolor": TINTA,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 24,              # espaço para o subtítulo entre o título e o gráfico
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRADE,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": TINTA_FRACA,
        "ytick.color": TINTA_FRACA,
        "xtick.labelcolor": TINTA_2,
        "ytick.labelcolor": TINTA_2,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2,
        "lines.solid_capstyle": "round",
        "font.size": 10,
        # usa a primeira fonte instalada da lista (Windows: Segoe UI; Mac: Helvetica)
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Segoe UI", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    })


def nova_figura(largura: float = 8.0, altura: float = 4.5, paineis: int = 1):
    """Cria a figura com o estilo do SCIC: um eixo, ou vários lado a lado (paineis > 1)."""
    aplicar_estilo()
    fig, eixos = plt.subplots(1, paineis, figsize=(largura, altura))
    for ax in ([eixos] if paineis == 1 else list(eixos)):
        ax.xaxis.set_major_formatter(FormatadorBR())
        ax.yaxis.set_major_formatter(FormatadorBR())
    return fig, eixos


def subtitulo(ax, texto: str) -> None:
    """Escreve uma linha de contexto entre o título e o gráfico."""
    ax.annotate(texto, xy=(0, 1), xycoords="axes fraction", xytext=(0, 7),
                textcoords="offset points", fontsize=9, color=TINTA_2, ha="left", va="bottom")


def salvar_figura(fig, nome_arquivo: str) -> Path:
    """Salva a figura em graficos_ou_imagens/ e devolve o caminho do PNG."""
    PASTA_GRAFICOS.mkdir(exist_ok=True)
    caminho = PASTA_GRAFICOS / nome_arquivo
    fig.savefig(caminho, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return caminho


def abrir_arquivo(caminho: Path) -> bool:
    """Abre um arquivo no programa padrão do sistema operacional.

    Devolve False quando não é possível abrir (por exemplo, num servidor sem tela).
    """
    try:
        sistema = platform.system()
        if sistema == "Windows":
            os.startfile(str(caminho))  # type: ignore[attr-defined]
        elif sistema == "Darwin":
            subprocess.Popen(["open", str(caminho)])
        else:
            subprocess.Popen(["xdg-open", str(caminho)], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
        return True
    except Exception:
        return False
