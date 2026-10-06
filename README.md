# SCIC – Sistema de Comunicação Interplanetária da Colônia Aurora Siger

FIAP · Ciência da Computação · Fase 6 · Atividade Integradora

| Integrante | RM |
|---|---|
| Mayara Luisa Vicente Rosa | RM571955 |
| Integrante 2 (preencher) | RM ______ |
| Integrante 3 (preencher) | RM ______ |
| Integrante 4 (preencher) | RM ______ |

**Arquivo principal:** `codigo_fonte.py`
**Vídeo de apresentação (YouTube, "Não listado"):** link em `link_video.txt`

---

## Objetivo

O SCIC é um protótipo em Python que organiza os dados operacionais e de comunicação da colônia
Aurora Siger, em Marte, e apoia decisões da equipe. Ele:

1. lê e limpa uma base simulada de 10 módulos durante 30 sóis (dias marcianos);
2. calcula indicadores de comunicação e os erros absoluto e relativo entre a latência prevista
   pela engenharia e a latência observada;
3. treina um modelo simples de regressão que prevê a latência dos enlaces internos e o avalia
   com MAE, MSE, RMSE, R², AIC e BIC;
4. prioriza os alertas críticos com um **max-heap** implementado do zero;
5. busca módulos, códigos de sensor, palavras de alerta e comandos por prefixo com uma **trie**;
6. relaciona o sistema com dispositivos de entrada e saída, bases numéricas e eletricidade
   (P = V × I e Lei de Ohm);
7. propõe um gerenciamento inteligente da comunicação: anomalias, eventos sistêmicos,
   manutenção preditiva, redundância de enlaces e eficiência energética.

Tudo é simulado, como o enunciado permite: não há sensores físicos, APIs externas nem hardware real.

## Menu do sistema

| Opção | O que faz | Enunciado |
|---|---|---|
| 1 | Carrega e limpa o CSV (relatório da limpeza) e cadastra novos registros | 5.1 |
| 2 | Consulta registros por módulo (nome parcial, sem acento), status e sol | 5.1 |
| 3 | Indicadores por módulo, erro absoluto e relativo, ponto flutuante (IEEE 754), Euler e Newton | 5.1, 5.2 |
| 4 | Previsão de latência para um módulo, carga, qualidade e perda informadas | 5.3 |
| 5 | Métricas do modelo contra dois baselines, AIC/BIC, validação cruzada e interpretação | 5.3 |
| 6 | Fila de prioridade de alertas com max-heap, top 5 e comparação com lista simples | 5.4 |
| 7 | Busca por prefixo com trie e comparação com busca linear | 5.5 |
| 8 | Código hexadecimal → binário → decimal, P = V × I, Lei de Ohm e dispositivos de E/S | 5.6 |
| 9 | Anomalias, evento sistêmico, manutenção preditiva, redundância e eficiência | 5.7 |
| 10 | Análise final com recomendações (salva em `graficos_ou_imagens/resumo_analise_final.txt`) | 6.1 |
| 0 | Sair | |

O menu aceita o número **ou o começo de um comando**: `pri` abre a opção 6, `trie` a 7,
`hard` a 8. O comando digitado é resolvido pela própria trie do sistema.

## Arquivos da entrega

```
SCIC_Aurora_Siger/
├── codigo_fonte.py          ARQUIVO PRINCIPAL: menu, entradas validadas e análise final
├── dados.py                 leitura, limpeza, consulta, cadastro e indicadores
├── erros.py                 erro absoluto e relativo, ponto flutuante, Euler e Newton
├── modelo.py                regressão linear, baselines, métricas, AIC/BIC, Grid Search
├── heap_alertas.py          critério de prioridade e max-heap (heapify-up e heapify-down)
├── trie_busca.py            trie com busca exata e por prefixo
├── hardware.py              bases numéricas, códigos de sensor, potência e Lei de Ohm
├── gestao_inteligente.py    anomalias, eventos, tendência, redundância e eficiência
├── graficos.py              estilo dos gráficos e números no padrão brasileiro
├── gerar_dados.py           gera a base simulada (seed 42)
├── dados_aurora_siger.csv   base de dados simulada (301 linhas, 16 colunas)
├── relatorio_tecnico.pdf    relatório técnico completo
├── README.md                este arquivo
├── requirements.txt         dependências
├── link_video.txt           link do vídeo no YouTube ("Não listado")
└── graficos_ou_imagens/     14 gráficos, exemplos de execução e resumo da análise final
```

Cada arquivo Python cuida de uma parte do enunciado. As funções dos módulos só calculam e devolvem
dados; quem imprime na tela é o `codigo_fonte.py`.

## Dependências

| Biblioteca | Uso no projeto | Versões testadas |
|---|---|---|
| NumPy | cálculos numéricos, Euler, ajuste de tendências | 1.23.5 a 2.5.3 |
| Pandas | leitura do CSV, limpeza, agrupamentos e tabelas | 1.5.3 a 3.0.5 |
| Matplotlib | gráficos salvos em PNG (sem abrir janelas) | 3.6.3 a 3.11.2 |
| scikit-learn | divisão treino/teste, regressão, métricas, validação cruzada, Grid Search | 1.2.2 a 1.9.1 |

Testado com Python 3.9, 3.11 e 3.13. Do Python padrão, o projeto usa `struct` e `decimal`
(IEEE 754), `heapq` (só para conferir o heap próprio), `json` (payload do n8n simulado),
`unicodedata` (busca sem acento), `random`, `time`, `gc` e `math`.

**Nenhuma biblioteca fora da lista do enunciado.** O Seaborn é permitido, mas não foi necessário.
O AIC e o BIC são calculados pela fórmula em NumPy, sem o `statsmodels` usado na aula.

## Como executar

```bash
# 1. dentro da pasta do projeto, instale as dependências (uma vez)
pip install -r requirements.txt

# 2. abra o menu interativo
python codigo_fonte.py

# 3. ou rode as 10 opções em sequência, sem perguntas (cerca de 10 segundos)
python codigo_fonte.py --demo
```

- No Windows, se `python` não for reconhecido, use `py -m pip install -r requirements.txt` e
  `py codigo_fonte.py`.
- Os gráficos são salvos em `graficos_ou_imagens/`; o menu pergunta se você quer abri-los.
- A opção 1 permite cadastrar um registro. Ele fica na memória e só vai para o CSV se você
  confirmar. Para restaurar a base original: `python gerar_dados.py`.

## Exemplos de execução

Saída completa do modo `--demo`: [`graficos_ou_imagens/exemplos_execucao.txt`](graficos_ou_imagens/exemplos_execucao.txt).

**Comando por prefixo e entradas inválidas**

```
Opção: abc
  Opção inválida. Digite um número de 0 a 10 ou o começo de um comando.
Opção: pri
  'pri' -> opção 6 (Priorizar alertas (heap))
Considerar alertas dos últimos quantos sóis [10]: 40
  Fora da faixa: use um valor de 1 a 30.
```

**Opção 6: os 5 alertas mais urgentes**

```
1. score 21,11 | sol 22 | Centro Médico (0x5101) | Queda de tensão no transmissor
2. score 20,60 | sol 27 | Centro Médico (0x5101) | Latência acima do previsto
3. score 19,93 | sol 22 | Controle Ambiental (0x7101) | Latência acima do previsto
4. score 19,77 | sol 21 | Controle Ambiental (0x7101) | Latência acima do previsto
5. score 19,28 | sol 25 | Controle Ambiental (0x7101) | Latência acima do previsto

Conferência com o heapq do Python (mesmo top 5): True
```

**Opção 7: busca por prefixo**

```
Prefixo para buscar (Enter para terminar): co
  módulos: Comando Central; Comunicação Orbital; Controle Ambiental
  comandos do menu: consultar -> opção 2
Prefixo para buscar (Enter para terminar): 0x5
  códigos de sensor: 0x5101 = Centro Médico
```

**Opção 8: código do sensor**

```
0x1201 em binário (cada dígito hex vira 4 bits): 0001 0010 0000 0001
em decimal (soma de potências de 2): 2^12 + 2^9 + 2^0 = 4096 + 512 + 1 = 4609
  tipo   = (4609 >> 12) & 0xF = 1 (comunicação)
  módulo = (4609 >> 8) & 0xF  = 2
  sensor = 4609 & 0xFF        = 1
```

## Dados

`dados_aurora_siger.csv` (UTF-8, separador vírgula, decimal com ponto) tem 301 linhas: 10 módulos
× 30 sóis, mais uma linha duplicada de propósito. A base foi gerada por `gerar_dados.py` com seed 42
e regras simples: a latência interna cresce com a carga da rede e a perda de pacotes; a do enlace
Terra–Marte é a distância dividida pela velocidade da luz. Também foram plantados problemas reais:
uma tempestade de poeira (sóis 12 e 13), a Estufa Hidropônica perdendo sinal, três quedas de tensão,
picos de retransmissão, quatro células vazias e a linha duplicada.

| Coluna | Tipo | Descrição |
|---|---|---|
| `id_registro` | inteiro | identificador do registro |
| `ciclo` | inteiro | sol (dia marciano) do registro, de 1 a 30 |
| `modulo` | texto | nome do módulo, como Centro Médico |
| `tipo_modulo` | texto | habitação, agricultura, comunicação, suporte médico… |
| `codigo_sensor` | texto (hex) | código de 16 bits: tipo, módulo e sensor (0x5101) |
| `distancia_km` | decimal | distância do enlace: até o hub (interno) ou até a Terra |
| `carga_rede_pct` | decimal | carga da rede no momento da medição (%) |
| `qualidade_sinal_pct` | decimal | qualidade do sinal recebido (%) |
| `perda_pacotes_pct` | decimal | pacotes perdidos e retransmitidos (%) |
| `tensao_v` | decimal | tensão no transmissor (V), nominal de 28 V |
| `corrente_a` | decimal | corrente do transmissor (A) |
| `latencia_prevista_ms` | decimal | estimativa de engenharia da latência (ms) |
| `latencia_observada_ms` | decimal | latência medida (ms) |
| `status` | texto | ativo, manutenção ou alerta |
| `prioridade_base` | inteiro | de 1 a 5 (5 = módulo que mantém vidas) |
| `mensagem_alerta` | texto | resumo do alerta, vazio quando ativo |

Potência, consumo, erros e faixas de erro são calculados pelo sistema, não digitados no CSV.

## Principais resultados

| Tema | Resultado |
|---|---|
| Limpeza | 301 linhas lidas, 299 válidas: 1 duplicada e 1 sem latência removidas; 3 vazios completados |
| Indicadores | 19,4% dos registros em alerta; menor disponibilidade na Estufa Hidropônica (63,3%) |
| Erros | a estimativa de engenharia erra 18,4% em média nos enlaces internos; no Terra–Marte, só 0,06% |
| Modelo | MAE de 7,9 ms, 52% menor que o da estimativa de engenharia; R² de 0,53 (0,95 sem 3 picos) |
| Heap | 24 alertas abertos; o mais urgente é a queda de tensão do Centro Médico no sol 22 |
| Trie | "co" traz 3 módulos e "com" traz 2; dezenas de vezes mais rápida que a busca linear |
| Hardware | 0x1201 = 4609 = comunicação, módulo 2, sensor 1; LED de alerta pede 150 Ω |
| Gestão | sinal da Estufa cruza 70% no sol 23, o que já era previsível no sol 10; enlace reserva leva a disponibilidade de 81,7% para 96,6% |

Os tempos dos benchmarks (heap × lista e trie × busca linear) variam conforme o computador.

## Decisões técnicas

- **Heap próprio**, guardado numa lista como na aula (pai em `(i-1)//2`, filhos em `2i+1` e `2i+2`).
  O `heapq` do Python só confere o resultado. Empates são desfeitos pela ordem de chegada.
- **Trie com dicionário de filhos** em vez do vetor de 26 letras da aula, porque os termos têm
  acentos, espaços, dígitos e "0x". O texto é normalizado, então "comunicacao" acha "Comunicação".
- **Enlace Terra–Marte fora do modelo**: sua latência está em minutos e depende da física
  (distância ÷ velocidade da luz). Ele entra na análise de erro relativo.
- **Escolha do modelo pelo BIC no treino** e avaliação no teste uma única vez, sem vazamento.
- **Gráficos salvos com o backend Agg**: o menu nunca trava esperando uma janela fechar.

## Referências

Materiais da Fase 6 da FIAP (enunciado e capítulos das disciplinas) e as fontes listadas no
`relatorio_tecnico.pdf`.
