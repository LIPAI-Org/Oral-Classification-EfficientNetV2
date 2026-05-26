# Classificação de Imagens Histológicas usando EfficientNet

<br><br>

## Autores

<div align="center">

| | | | |
|:---:|:---:|:---:|:---:|
| [![Ana Clara](https://github.com/anaclaramcarvalho.png?size=80)](https://github.com/anaclaramcarvalho) | [![Diego](https://github.com/diegodemiranda.png?size=80)](https://github.com/diegodemiranda) | [![Sthephanny](https://github.com/sthecss.png?size=80)](https://github.com/sthecss) | [![Victor](https://github.com/VictorBertolini.png?size=80)](https://github.com/VictorBertolini) |
| [Ana Clara](https://github.com/anaclaramcarvalho) | [Diego](https://github.com/diegodemiranda) | [Sthephanny](https://github.com/sthecss) | [Victor](https://github.com/VictorBertolini) |

</div>

<br><br>

## Sumário

- [Descrição dos Datasets](#descrição-dos-datasets)
- [Arquiteturas Utilizadas](#arquiteturas-utilizadas)
- [Modos de Treinamento](#modos-de-treinamento)
- [Estratégia de Data Augmentation](#estratégia-de-data-augmentation)
- [Desenho Experimental](#desenho-experimental)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Como Executar](#como-executar)
- [Limitações Conhecidas](#limitações-conhecidas)
- [Estrutura de Diretórios](#estrutura-de-diretórios)

<br><br>

## Descrição dos Datasets

### Dataset 1 — OralEpitheliumDB

Desenvolvido pela Universidade Federal de Uberlândia (UFU). Focado em Displasia Epitelial Oral (OED), com imagens microscópicas de tecido bucal com potencial de transformação maligna. As imagens foram coletadas de 30 línguas de camundongo, com núcleos celulares marcados manualmente por especialista e rótulos avaliados e validados por patologista.

- **Total de imagens:** 456
- **Tarefa:** Classificação binária
- **Subconjunto utilizado:** `Original ROI Images`
- **Fonte:** https://github.com/LIPAI-Org/OralEpitheliumDB_Dataset

**Classes e divisão dos splits:**

| Split      | healthy | severe | Total |
|------------|--------:|-------:|------:|
| Treino     |      72 |     72 |   144 |
| Validação  |      19 |     19 |    38 |
| Teste      |      23 |     23 |    46 |
| **Total**  |  **114**|**114** | **228**|

<br>

### Dataset 2 — NDB-UFES

Desenvolvido pela Universidade Federal do Espírito Santo (UFES). Contém imagens histopatológicas de biópsias de pacientes atendidos pelo projeto de Diagnóstico Oral (NDB) entre 2010 e 2021. Além das imagens, o dataset inclui dados sociodemográficos (gênero, idade, cor da pele) e clínicos (tabagismo, consumo de álcool, exposição solar) — utilizados apenas como contexto, não no treinamento.

- **Total de imagens:** 237
- **Tarefa:** Classificação multiclasse
- **Fonte:** https://data.mendeley.com/datasets/bbmmm4wgr8/4

**Classes e divisão dos splits:**

| Split      | Leukoplakia w/ dysplasia | Leukoplakia w/o dysplasia | OSCC | Total |
|------------|-------------------------:|--------------------------:|-----:|------:|
| Treino     |                       59 |                        39 |   61 |   159 |
| Validação  |                       15 |                         9 |   15 |    39 |
| Teste      |                       15 |                         9 |   15 |    39 |
| **Total**  |                   **89** |                    **57** | **91** | **237** |

> Os splits de ambos os datasets são fixos e compartilhados entre todos os grupos para permitir comparação direta dos resultados.

---

<br><br>

## Arquiteturas Utilizadas

Ambas as arquiteturas são carregadas via biblioteca [TIMM](https://github.com/huggingface/pytorch-image-models):

- [EfficientNetV2-B0](https://huggingface.co/timm/tf_efficientnetv2_b0.in1k) (`tf_efficientnetv2_b0.in1k`)
- [EfficientNetV2-B1](https://huggingface.co/timm/tf_efficientnetv2_b1.in1k) (`tf_efficientnetv2_b1.in1k`)

Cada modelo é composto por um **backbone** (responsável pela extração automática de características visuais) e uma **camada classificadora fully connected** (responsável pela predição das classes). Os pesos pré-treinados são baixados automaticamente pelo `timm` na primeira execução — nenhum passo manual é necessário.

---

<br><br>

## Modos de Treinamento

Para cada arquitetura e dataset, três modos de treinamento foram avaliados:

| Modo | Nome | Descrição                                                                                           |
|------|------|-----------------------------------------------------------------------------------------------------|
| `FS` | From Scratch | Todos os pesos inicializados aleatoriamente; treinamento do zero sem treinamento prévio             |
| `PT-FC` | Pré-treinado + Backbone Congelado | Pesos pré-treinados carregados; backbone congelado; apenas a camada classificadora final é treinada |
| `PT-ALL` | Pré-treinado + Fine-tuning Completo | Pesos pré-treinados carregados; todas as camadas — backbone e classificador — são treinadas         |

---

<br><br>

## Estratégia de Data Augmentation

As transformações de augmentation são aplicadas **exclusivamente no conjunto de treino**. Validação e teste usam apenas redimensionamento e normalização.

**Sem augmentation:**
- `Resize(224, 224)` + Normalização ImageNet

**Com augmentation (aplicadas sobre o conjunto de treino):**
- `Resize(224, 224)`
- `RandomHorizontalFlip` com probabilidade de 50%
- `RandomRotation` sorteando entre 0°, 90°, 180° ou 270° com probabilidade igual (25% cada)
- Normalização ImageNet

> Normalização padrão ImageNet: `mean=(0.485, 0.456, 0.406)`, `std=(0.229, 0.224, 0.225)`

---

<br><br>

## Desenho Experimental

```
2 arquiteturas × 3 modos × 2 condições de augmentation × 2 datasets × 3 seeds = 72 execuções
```

**Seeds utilizadas:** `42`, `123`, `2025`

Para cada configuração são executadas 3 repetições (uma por seed). As métricas finais reportadas são a **média** e o **desvio padrão** das 3 repetições.

**Protocolo de treinamento:**

| Parâmetro | Valor |
|-----------|-------|
| Entrada | 224 × 224 px |
| Normalização | ImageNet |
| Épocas | 50 |
| Otimizador | Adam |
| Taxa de aprendizado | 1e-4 |
| Batch size | 32 |
| Função de perda | Cross Entropy |
| Critério de seleção | Maior acurácia de validação |

---

<br><br>

## Requisitos

- Python 3.10+
- Git
- VS Code com a extensão [Python](https://marketplace.visualstudio.com/items?itemName=ms-python.python) e [Jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)

---

<br><br>

## Instalação


### Linux/macOS

**Terminal do PC:**
```bash
git clone https://github.com/LIPAI-Org/Oral-Classification-EfficientNetV2.git
cd Oral-Classification-EfficientNetV2
python3 -m venv .venv
source .venv/bin/activate
```

**Terminal do VS Code** (`Ctrl+` `):
```bash
pip install timm pandas torch torchvision Pillow numpy scikit-learn thop matplotlib seaborn jupyter ipykernel requests tqdm
```

<br>

### Windows

**Terminal do PC:**
```bash
git clone https://github.com/LIPAI-Org/Oral-Classification-EfficientNetV2.git
cd Oral-Classification-EfficientNetV2
python -m venv .venv
.venv\Scripts\activate
```

**Terminal do VS Code** (`Ctrl+` `):
```bash
pip install timm pandas torch torchvision Pillow numpy scikit-learn thop matplotlib seaborn jupyter ipykernel requests tqdm
```

<br><br>

> **VS Code:** após criar o `.venv`, pressione `Ctrl+Shift+P` → `Python: Select Interpreter` e selecione `.venv/bin/python` (Linux) ou `.venv\Scripts\python.exe` (Windows) para garantir que o ambiente virtual seja usado nos notebooks.

---

<br><br>

## Como Executar

### 1. Baixar os datasets

```bash
python scripts/setup_datasets.py
```

Se o download automático falhar, baixe manualmente:
- **Dataset 1:** https://zenodo.org/records/17693395/files/Original%20ROI%20images.zip?download=1 → extrair em `datasets/`
- **Dataset 2:** https://data.mendeley.com/datasets/bbmmm4wgr8/4 → extrair em `datasets/images/`

Certifique-se de que os arquivos `.csv` dos splits estejam em `datasets/`.

<br>

### 2. Rodar o treinamento

```bash
jupyter nbconvert --to notebook --execute notebooks/main.ipynb
```

Ou abra `notebooks/main.ipynb` diretamente no Jupyter/VS Code e execute célula por célula.

Ao final, o arquivo `results.csv` será gerado na raiz do projeto com os resultados das 72 execuções.

<br>

### 3. Gerar métricas e figuras

```bash
jupyter nbconvert --to notebook --execute notebooks/metrics.ipynb
```

Ou abra `notebooks/metrics.ipynb` e execute célula por célula.

Todos os gráficos e PDFs vetoriais serão salvos em `figures/`.

---

<br><br>

## Limitações Conhecidas

- Os experimentos foram executados apenas em CPU (i7-12700, 16 GB RAM), o que aumenta consideravelmente o tempo de treinamento das 72 execuções.
- O NDB-UFES apresenta desbalanceamento de classes (especialmente `Leukoplakia without dysplasia`), o que pode afetar métricas como F1-score macro.
- O limite de 50 épocas pode ser insuficiente para convergência plena no modo From Scratch, especialmente no dataset multiclasse.
- Possível overfitting nos modelos fine-tuned no OralEpitheliumDB, dado o tamanho reduzido do dataset (228 imagens).

---

<br><br>

> [!IMPORTANT]
> As pastas `checkpoints/`, `histories/`, `confusion_matrices/` não estão incluídos no repositório — 
> eles são gerados automaticamente ao executar o `main.ipynb`. Já o arquivo `results.csv` está na raiz do projeto para visto.

<br>

## Estrutura de Diretórios:

```
.
├── requirements.txt                            ← dependências Python
├── README.md
│
├── scripts/
│   └── setup_datasets.py                       ← baixa e extrai os datasets
│
├── notebooks/
│   ├── main.ipynb                              ← treinamento das 72 execuções
│   └── metrics.ipynb                           ← geração de métricas e gráficos
│
├── datasets/
│   ├── manifest_split_oralepitheliumdb.csv     ← splits fixos Dataset 1
│   ├── manifest_split_multiclass_NDB-UFES.csv  ← splits fixos Dataset 2
│   ├── Original ROI images/                    ← Dataset 1 (baixado via setup)
│   │   ├── healthy/
│   │   └── severe/
│   └── images/                                 ← Dataset 2 (baixado via setup)
│
├── figures/                                    ← gráficos gerados pelo metrics.ipynb
│
├── results.csv                                 ← gerado após o treinamento (72 linhas)
├── checkpoints/                                ← pesos do melhor modelo por execução
├── histories/                                  ← histórico loss/acurácia por época
└── confusion_matrices/                         ← matrizes de confusão por execução
```
