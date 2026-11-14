# Fuzzy Sentic Analysis: Baseline & Modern NLP Framework

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Fuzzy Logic](https://img.shields.io/badge/Theory-Fuzzy%20Sets%20%26%20Sentic%20Computing-purple.svg)]()

Este repositório contém uma implementação modular de referência do framework proposto por **Loia & Senatore (2014)** (*"A fuzzy-oriented sentic analysis to capture the human emotion in Web-based content"*), acompanhada de uma arquitetura extensível projetada para modernizar suas etapas linguísticas e matemáticas utilizando **NLP moderno leve** e **operadores nebulosos avançados**.

O projeto foi concebido seguindo boas práticas de engenharia de software (Padrão *Strategy*, Separação de Responsabilidades e Inversão de Dependência) e otimizado para execução eficiente em hardware convencional (CPU ou GPUs de entrada), sem a necessidade de grandes modelos generativos (LLMs) pesados.

---

## 🎯 Objetivos Científicos e Arquiteturais

1. **Reprodutibilidade do Baseline (Loia & Senatore, 2014):** Reproduz fielmente o pipeline clássico baseado em *POS Tagging*, *WordNet*, *SentiWordNet*, similaridade semântica de *Wu & Palmer*, heurísticas de antônimos na *Hourglass of Emotions* e modificadores linguísticos nebulosos (intensificadores, negadores e combinadores).
2. **Extensibilidade para Pesquisa (Modernização):** Fornece um contrato abstrato (`SenticPipelineInterface`) e um esqueleto estrutural (`NeoSenticFuzzy`) para substituir gargalos léxicos antigos por técnicas modernas leves:
   * **Análise Sintática:** Árvores de Dependência (*Dependency Parsing* via `spaCy`) no lugar de padrões fixos de etiquetas POS.
   * **Representação Semântica:** *Embeddings* densos contextuais (*Sentence-Transformers*) e similaridade cosseno no lugar de métricas baseadas em grafos do *WordNet*.
   * **Modelagem Nebulosa Refinada:** Modificadores linguísticos paramétricos (*Hedges* de Zadeh $x^\alpha$), complementos assimétricos de Sugeno e agregação via operadores **OWA** (*Ordered Weighted Averaging*) ou T-conormas paramétricas.
3. **Avaliação Comparativa Justa:** Pipeline de benchmark agnóstico ao modelo, permitindo comparar lado a lado o baseline clássico e novas abordagens sob exatamente as mesmas métricas (*Precision*, *Recall* e *F-measure* por limiar $t$, replicando a Tabela 10 do artigo original).

---

## 📁 Estrutura do Repositório

```text
fuzzy_sentic_analysis/
├── data/                                 # Datasets brutos (ex: Sentiment140 CSV)
├── results/                              # Saídas de experimentos (CSVs de métricas, tabelas)
├── src/
│   └── fuzzy_sentic/
│       ├── __init__.py
│       ├── config.py                     # Constantes, dimensões da Hourglass e listas léxicas
│       ├── interfaces.py                 # Contrato abstrato (SenticPipelineInterface)
│       ├── models/
│       │   ├── __init__.py
│       │   ├── baseline.py               # Implementação fiel de Loia & Senatore (2014)
│       │   └── neosentic.py                 # Esqueleto da nossa abordagem
│       └── evaluation/
│           ├── __init__.py
│           ├── data_loader.py            # Leitura e amostragem estratificada dos datasets
│           └── metrics.py                # Cálculo de Precision, Recall, F1
├── run_demo.py                           # Execução rápida e inspeção da matriz emocional (6x4)
├── run_benchmark.py                      # Avaliação quantitativa comparativa em lote
├── requirements.txt                      # Dependências do projeto
└── README.md                             # Documentação do repositório
```

---

## ⚙️ Instalação e Configuração

### 1. Clonar o repositório e criar o ambiente virtual

```bash
git clone <url-do-seu-repositorio>
cd fuzzy_sentic_analysis

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

*(Nota: Na primeira execução do baseline, os recursos necessários do `nltk` — `wordnet`, `sentiwordnet`, `punkt` e `averaged_perceptron_tagger` — serão baixados automaticamente.)*

### 3. Preparar o Dataset para Benchmark (Opcional)

Para executar a avaliação quantitativa da Seção 6.1 do artigo (Tabela 10):
1. Baixe o dataset **Sentiment140** (`training.1600000.processed.noemoticon.csv`) no Kaggle ou repositório oficial.
2. Coloque o arquivo CSV dentro da pasta `data/`:
   ```text
   data/training.1600000.processed.noemoticon.csv
   ```

---

## 🚀 Como Utilizar

### 1. Demonstração Qualitativa (`run_demo.py`)
Executa a análise de sentimento e extrai a matriz emocional completa ($6 \text{ níveis sentic} \times 4 \text{ dimensões da Hourglass of Emotions}$) para sentenças de exemplo:

```bash
python run_demo.py
```

**Exemplo de saída:**
```text
Modelo: LoiaSenatoreBaseline
Texto analisado: 'He is happy to go home. However, the weather is not very good.'
Polaridade Global (Eq. 5): 0.4525

Matriz Emocional Final (6 Níveis Sentic x 4 Dimensões):
Nível    | Pleasantness | Attention  | Sensitivity | Aptitude  
--------------------------------------------------------------
+3       | 0.750        | 0.000      | 0.000       | 0.800     
+2       | 0.800        | 0.000      | 0.000       | 0.570     
+1       | 0.000        | 0.000      | 0.000       | 0.570     
-1       | 0.000        | 0.000      | 0.750       | 0.000     
-2       | 0.000        | 0.000      | 0.000       | 0.000     
-3       | 0.000        | 0.850      | 0.000       | 0.000     
```

### 2. Benchmark Quantitativo (`run_benchmark.py`)
Avalia os modelos registrados sobre uma amostra do *Sentiment140* variando os limiares $t \in \{\pm 0.1, \pm 0.2, \pm 0.4, \pm 0.6\}$ e exporta os resultados para `results/`:

```bash
python run_benchmark.py
```

### 3. Implementando e Integrando a Versão Modernizada
Para adicionar sua nova abordagem:
1. Implemente os métodos definidos em `src/fuzzy_sentic/models/modern.py` na classe `ModernFuzzySentic`.
2. Certifique-se de que `analyze_text(text)` retorne a tupla `(total_polarity, text_emo_table)` no formato `(float, np.ndarray[6, 4])`.
3. Descomente a importação e instanciação de `ModernFuzzySentic` nos scripts `run_demo.py` e `run_benchmark.py` para gerar automaticamente os comparativos.

---

## 🧩 Mapeamento entre o Artigo Base e o Código

| Seção do Artigo (Loia & Senatore, 2014) | Descrição Conceitual | Módulo Correspondente |
| :--- | :--- | :--- |
| **Seção 4** | *Hourglass of Emotions* (4 dimensões $\times$ 6 níveis) e léxicos | `src/fuzzy_sentic/config.py` |
| **Seção 5.1** | *POS Tagging*, lematização e substantivação no WordNet | `baseline.py` (`phase1_pos_tagging_and_filtering`) |
| **Seção 5.2** | Polaridade (Eq. 1) e Tabela de Similaridade Emocional | `baseline.py` (`compute_word_polarity`, `compute_emotional_similarity_table`) |
| **Seção 5.3** | Padrões sintáticos (Tabela 5) e Modificadores Fuzzy (Eqs. 2–4) | `baseline.py` (`phase3_pattern_analysis`, `_fuzzy_*`) |
| **Seções 5.4 & 5.5** | Agregação por frase ($\alpha$-cut) e texto completo (Eq. 5) | `baseline.py` (`phase4_phrase_analysis`, `analyze_text`) |
| **Seção 6.1** | Avaliação de *Precision*, *Recall* e *F-measure* (Tabela 10) | `src/fuzzy_sentic/evaluation/metrics.py` |

---

## 📚 Referências Principais

* **Artigo Base:** Loia, V., & Senatore, S. (2014). *A fuzzy-oriented sentic analysis to capture the human emotion in Web-based content*. Knowledge-Based Systems, 58, 75–85.
* **Modelo Emocional:** Cambria, E., Livingstone, A., & Hussain, A. (2012). *The Hourglass of Emotions*. In Cognitive Behavioural Systems (pp. 144–157). Springer.
* **Operadores de Agregação Nebulosos:** Yager, R. R. (1988). *On ordered weighted averaging aggregation operators in multicriteria decisionmaking*. IEEE Transactions on Systems, Man, and Cybernetics, 18(1), 183–190.