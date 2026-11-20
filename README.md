# Fuzzy Sentic Analysis: Baseline & Modern NLP Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Fuzzy Logic](https://img.shields.io/badge/Theory-Fuzzy%20Sets%20%26%20Sentic%20Computing-purple.svg)]()

Este repositório contém uma implementação modular de referência do framework proposto por **Loia & Senatore (2014)** (*"A fuzzy-oriented sentic analysis to capture the human emotion in Web-based content"*), acompanhada de uma arquitetura extensível projetada para modernizar suas etapas linguísticas e matemáticas utilizando **NLP moderno leve** e **operadores nebulosos avançados**.

O projeto foi concebido seguindo boas práticas de engenharia de software (Padrão *Strategy*, Separação de Responsabilidades e Inversão de Dependência) e otimizado para execução eficiente em hardware convencional (CPU ou GPUs de entrada), sem a necessidade de grandes modelos generativos (LLMs) pesados.

---

## 🎯 Objetivos Científicos e Arquiteturais

1. **Reprodutibilidade do Baseline (Loia & Senatore, 2014):** Reproduz fielmente o pipeline clássico baseado em *POS Tagging*, *WordNet*, *SentiWordNet*, similaridade semântica de *Wu & Palmer*, heurísticas de antônimos na *Hourglass of Emotions* e modificadores linguísticos nebulosos (intensificadores, negadores e combinadores).
2. **Extensibilidade para Pesquisa (Modernização):** Fornece um contrato abstrato (`SenticPipelineInterface`) e a implementação `NeoSenticFuzzy`, que substitui gargalos léxicos antigos por técnicas modernas leves:
   * **Análise Sintática:** Árvores de Dependência (*Dependency Parsing* via `spaCy`) no lugar de padrões fixos de etiquetas POS.
   * **Representação Semântica:** *Embeddings* densos contextuais (*Sentence-Transformers*) e similaridade cosseno no lugar de métricas baseadas em grafos do *WordNet*.
   * **Modelagem Nebulosa Refinada:** Modificadores linguísticos paramétricos (*Hedges* de Zadeh $x^\alpha$), complementos assimétricos de Sugeno e agregação via operadores **OWA** (*Ordered Weighted Averaging*) ou T-conormas paramétricas.
3. **Avaliação Comparativa Justa:** Pipeline de benchmark agnóstico ao modelo, permitindo comparar lado a lado o baseline clássico e novas abordagens sob exatamente as mesmas métricas (*Precision*, *Recall* e *F-measure* por limiar $t$, com o protocolo da Tabela 10 do artigo, aplicado sempre aos mesmos dados; ver [`docs/COMP.md`](docs/COMP.md)).

---

## 📁 Estrutura do Repositório

```text
NeoSenticFuzzy/
├── data/                                 # Sentiment140 (não versionado)
├── docs/
│   └── COMP.md                           # Protocolo de comparação (mesmos dados)
├── results/                              # Tabelas de métricas e contagens da Figura 2 (CSV)
├── resultados_execucao/                  # Gráficos de comparação (PNG)
├── src/
│   └── fuzzy_sentic/
│       ├── config.py                     # Hourglass, léxicos e parâmetros do Neo
│       ├── interfaces.py                 # Contrato abstrato (SenticPipelineInterface)
│       ├── text_utils.py                 # Normalização de tweets compartilhada pelos modelos
│       ├── conversation.py               # Classificação de uma conversa (usada pelo chat)
│       ├── models/
│       │   ├── baseline.py               # Implementação fiel de Loia & Senatore (2014)
│       │   └── neosentic.py              # NeoSenticFuzzy (nossa abordagem)
│       └── evaluation/
│           ├── data_loader.py            # Amostra estratificada e conjunto de teste do Sentiment140
│           ├── metrics.py                # Precision, Recall e F-measure por limiar (Tabela 10)
│           └── emotions.py               # Contagem de emoções por nível sentic (Figura 2)
├── tests/                                # Testes (inclui os valores publicados no artigo)
├── run_demo.py                           # Exemplo do artigo nos dois modelos
├── run_chat.py                           # Chat no terminal com classificação em tempo real
├── pytest.ini
└── requirements.txt
```

---

## ⚙️ Instalação e Configuração

### 1. Clonar o repositório e criar o ambiente virtual

```bash
git clone git@github.com:gdac7/NeoSenticFuzzy.git
cd NeoSenticFuzzy

python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate
```

### 2. Instalar as dependências

```bash
pip install -r requirements.txt
```

O `requirements.txt` instala o PyTorch em versão CPU e o modelo `en_core_web_sm` do spaCy. Na primeira execução, os recursos do `nltk` (`wordnet`, `sentiwordnet`, `punkt_tab`, tagger) e o modelo `all-MiniLM-L6-v2` do Sentence-Transformers são baixados automaticamente.

### 3. Dataset (Sentiment140)

Os resultados usam o **Sentiment140**. Para tê-lo localmente, baixe e extraia em `data/`:

```bash
mkdir -p data && cd data
curl -LO https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
unzip trainingandtestdata.zip && rm trainingandtestdata.zip
```

São usados `training.1600000.processed.noemoticon.csv` (amostra de 5000 tweets) e `testdata.manual.2009.06.14.csv` (359 tweets positivos e negativos anotados manualmente pelos autores do Sentiment140; os neutros são descartados).

---

## 🚀 Como Utilizar

### 1. Demonstração Qualitativa (`run_demo.py`)
Executa a análise de sentimento e extrai a matriz emocional completa ($6 \text{ níveis sentic} \times 4 \text{ dimensões da Hourglass of Emotions}$) para sentenças de exemplo:

```bash
python run_demo.py
```

**Exemplo de saída** (frase do exemplo do artigo, Seções 5.1–5.4):
```text
Modelo: LoiaSenatoreBaseline
Texto analisado: 'He is happy to go home.'
Polaridade Global: 0.2502

Matriz Emocional Final (6 Níveis Sentic x 4 Dimensões):
Nível    | Pleasantness | Attention  | Sensitivity | Aptitude
--------------------------------------------------------------
+3       | 0.889        | 0.000      | 0.000       | 0.706
+2       | 0.824        | 0.000      | 0.000       | 0.625
+1       | 0.556        | 0.000      | 0.000       | 0.625
-1       | 0.000        | 0.000      | 0.778       | 0.000
-2       | 0.000        | 0.000      | 0.000       | 0.000
-3       | 0.000        | 0.750      | 0.000       | 0.000
```

As emoções ativas são as mesmas da Tabela 8 do artigo (Ecstasy, Joy, Amazement, Apprehension, Admiration, Trust, Acceptance), mais Serenity. A diferença vem de `happy` ser ligado a `happiness.n.01` em vez de `happiness#n#2` (veja "Desvios do artigo").

### 2. Chat em tempo real (`run_chat.py`)
Conversa no terminal em que cada mensagem enviada alterna entre **Pessoa A** e **Pessoa B**. O painel da direita mostra, a cada mensagem, a classificação da conversa (positiva, negativa ou neutra, com limiar 0.1), a polaridade de cada pessoa e da última mensagem, as emoções dominantes da Hourglass e o tempo de processamento de cada atualização (útil para comparar os dois modelos rodando em terminais separados).

```bash
python run_chat.py                    # NeoSenticFuzzy
python run_chat.py --modelo baseline  # Loia & Senatore (2014)
```

`Ctrl+L` limpa a conversa e `Ctrl+Q` sai. Os dois modelos só entendem **inglês** (SentiWordNet, WordNet, spaCy e MiniLM são recursos em inglês); frases em português dão resultados sem sentido.

### 3. Testes

```bash
pytest
```

Os testes do baseline conferem os valores publicados no artigo: Eq. 1, Eqs. 2–4, Tabelas 3, 4, 6, 7 e 8, e os padrões da Tabela 5.

---

## 🧠 NeoSenticFuzzy: o que muda em relação ao artigo

| Etapa | Baseline (Loia & Senatore, 2014) | NeoSenticFuzzy |
| :--- | :--- | :--- |
| Estrutura da frase | Padrões fixos de POS entre palavras vizinhas (Tabela 5) | Árvore de dependências do spaCy (negação e intensificadores alcançam a palavra certa, mesmo à distância) |
| Polaridade da palavra | Eq. 1: média do SentiWordNet sobre todos os sentidos | SentiWordNet com o peso de cada sentido ajustado pela semelhança entre a definição e a frase (Sentence-Transformers); gírias fora do dicionário recebem polaridade por proximidade com protótipos |
| Emoções | Wu-Palmer no grafo do WordNet | Cosseno entre embeddings da palavra e dos 24 rótulos da Hourglass |
| Intensificadores | Eq. 2, igual para todos | Hedges de Zadeh com força por palavra (*extremely* > *very* > *really*; *slightly* atenua) |
| Negação | Eqs. 3 e 4 | Negação assimétrica de Sugeno ("not good" negativo, "not bad" levemente positivo) |
| Agregação da polaridade | Média aritmética | IOWA induzida pela força da opinião |
| União das emoções | Máximo | T-conorma de Yager |

**IOWA** (*Induced Ordered Weighted Averaging*; Yager & Filev, 1999) ordena as polaridades pela força (|p|) e aplica pesos por posição, vindos do quantificador Q(r) = r^β (β = 0.5): a opinião mais forte pesa mais, sem ignorar as demais. Por exemplo, com as polaridades de *happy* (0.70), *home* (0.01) e *go* (0.004), a média dá 0.24 (o "happy" é diluído) e a IOWA dá 0.41. Com β = 1 ela vira a média comum. Os parâmetros do Neo ficam em `config.py` (`NEO_*`) e foram fixados a priori, sem ajuste nos dados de avaliação.

## 📊 Resultados

Todos os modelos são avaliados **nos mesmos dados**: nossa amostra estratificada de 5000 tweets do Sentiment140 e o teste anotado manualmente (359 tweets), com as métricas da Seção 6.1 do artigo. A Tabela 10 do artigo **não** entra na comparação, porque usa outros tweets e outro gabarito; o motivo está em [`docs/COMP.md`](docs/COMP.md).

F-measure na amostra de 5000 tweets (P, R e F completos em `results/table10_*.csv`; gráficos em `resultados_execucao/`):

| Limiar | Classe | Baseline original | Baseline corrigido | NeoSenticFuzzy |
| :--- | :--- | ---: | ---: | ---: |
| ±0.1 | Positive | 0.433 | 0.452 | **0.579** |
| ±0.1 | Negative | 0.315 | 0.341 | **0.481** |
| ±0.2 | Positive | 0.276 | 0.303 | **0.452** |
| ±0.2 | Negative | 0.158 | 0.182 | **0.357** |
| ±0.4 | Positive | 0.089 | 0.107 | **0.259** |
| ±0.4 | Negative | 0.047 | 0.041 | **0.144** |
| ±0.6 | Positive | 0.047 | 0.060 | **0.071** |
| ±0.6 | Negative | 0.013 | 0.014 | **0.043** |

No teste manual, o F em ±0.1 vai de 0.673 / 0.419 (baseline corrigido) para 0.737 / 0.669 (Neo), em positivos / negativos.

O método do artigo produz polaridades pequenas: a Eq. 1 faz média sobre todos os sentidos da palavra, e a média da frase inclui palavras quase neutras (*"He is happy to go home"* dá 0.25). Por isso poucos tweets passam de ±0.4. O NeoSenticFuzzy ataca exatamente esses pontos (sentido escolhido pelo contexto e agregação IOWA) e melhora o F em todos os limiares, com precisão parecida.

**Figura 2.** No baseline, com corte 0.2 praticamente todos os tweets ativam todas as emoções: substantivos sem afinidade emocional mantêm a tabela crua (≈0.2), como `home` na Tabela 7, e palavras afetivas ficam acima de 0.5 em quase todos os rótulos. No Neo, as emoções aparecem em menos tweets, mas do lado certo: Joy em 523 tweets positivos contra 44 negativos, e Sadness em 74 contra 235 (`results/fig2_*.csv`). Como o Sentiment140 não tem rótulos de emoção, isso mostra coerência, não acerto.

## ↔️ Desvios do artigo

* **WordNet-Affect** (indisponível: o site da FBK está fora do ar): as categorias afetivas são os substantivos do WordNet 3.0 descendentes de `feeling.n.01`, ligados à palavra por derivação, atributo ou pertainym. Por isso `happy` vai para `happiness.n.01`, e não para `happiness#n#2`. Isso só afeta a matriz emocional, não a polaridade.
* **Wu-Palmer**: a JWSL usada no artigo equivale à fórmula do NLTK com a profundidade deslocada em +2 (`wup_jwsl`), o que reproduz a Tabela 3 com erro médio de 0.006.
* **Heurística dos opostos**: a regra que reproduz as Tabelas 4 e 7 é âncora = max(v, 1 − v), oposto = 1 − âncora, e palavras sem afinidade emocional (máximo < 0.5) mantêm a tabela crua.
* **Polaridade de padrões** limitada a [−1, 1] (o artigo soma sem limitar).
* **Eq. 4** aplicada literalmente: "not very happy" (0.883) fica mais positivo que "happy" (0.847).
* **Normalização de tweets** (entidades HTML, URLs, @menções, #, emoticons) igual para os dois modelos.

---

## 🧩 Mapeamento entre o Artigo Base e o Código

| Seção do Artigo (Loia & Senatore, 2014) | Descrição Conceitual | Módulo Correspondente |
| :--- | :--- | :--- |
| **Seção 4** | *Hourglass of Emotions* (4 dimensões $\times$ 6 níveis) e léxicos | `src/fuzzy_sentic/config.py` |
| **Seção 5.1** | *POS Tagging*, lematização, desambiguação e substantivação no WordNet | `baseline.py` (`phase1_pos_tagging_and_filtering`, `assign_synsets`) |
| **Seção 5.2** | Polaridade (Eq. 1) e Tabela de Similaridade Emocional | `baseline.py` (`compute_word_polarity`, `compute_emotional_similarity_table`) |
| **Seção 5.3** | Padrões sintáticos (Tabela 5) e Modificadores Fuzzy (Eqs. 2–4) | `baseline.py` (`phase3_pattern_analysis`, `_pattern_unit`, `_fuzzy_*`) |
| **Seções 5.4 & 5.5** | Agregação por frase ($\alpha$-cut) e texto completo (Eq. 5) | `baseline.py` (`phase4_phrase_analysis`, `analyze_text`) |
| **Seção 6.1** | Avaliação de *Precision*, *Recall* e *F-measure* (Tabela 10) | `src/fuzzy_sentic/evaluation/metrics.py` |
| **Seção 6.2** | Emoções por nível sentic nos tweets positivos e negativos (Figura 2) | `src/fuzzy_sentic/evaluation/emotions.py` |

---

## 📚 Referências Principais

* **Artigo Base:** Loia, V., & Senatore, S. (2014). *A fuzzy-oriented sentic analysis to capture the human emotion in Web-based content*. Knowledge-Based Systems, 58, 75–85.
* **Modelo Emocional:** Cambria, E., Livingstone, A., & Hussain, A. (2012). *The Hourglass of Emotions*. In Cognitive Behavioural Systems (pp. 144–157). Springer.
* **Dataset:** Go, A., Bhayani, R., & Huang, L. (2009). *Twitter Sentiment Classification using Distant Supervision*. CS224N Project Report, Stanford.
* **Operadores de Agregação Nebulosos:** Yager, R. R. (1988). *On ordered weighted averaging aggregation operators in multicriteria decisionmaking*. IEEE Transactions on Systems, Man, and Cybernetics, 18(1), 183–190.
* **IOWA:** Yager, R. R., & Filev, D. P. (1999). *Induced ordered weighted averaging operators*. IEEE Transactions on Systems, Man, and Cybernetics, Part B, 29(2), 141–150.
