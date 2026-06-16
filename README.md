# Classificação de Imagens Histológicas Orais usando EfficientNetV2 & Grad-CAM

Este projeto visa a classificação e análise de imagens de histopatologia oral utilizando a arquitetura de Deep Learning **EfficientNetV2**, acompanhada de mapas de ativação clássicos (**Grad-CAM**) para explicabilidade do modelo (interpretação das regiões visuais que motivaram a classificação tecidual).

<br><br>

## Autores

<div align="center">

| | | | |
|:---:|:---:|:---:|:---:|
| [![Ana Clara](https://github.com/anaclaramcarvalho.png?size=80)](https://github.com/anaclaramcarvalho) | [![Diego](https://github.com/diegodemiranda.png?size=80)](https://github.com/diegodemiranda) | [![Sthephanny](https://github.com/sthecss.png?size=80)](https://github.com/sthecss) | [![Victor](https://github.com/VictorBertolini.png?size=80)](https://github.com/VictorBertolini) |
| [Ana Clara](https://github.com/anaclaramcarvalho) | [Diego](https://github.com/diegodemiranda) | [Sthephanny](https://github.com/sthecss) | [Victor](https://github.com/VictorBertolini) |

</div>

<br><br>


Para garantir a reprodutibilidade, o projeto está preparado para rodar nativamente no [**Kaggle**](https://www.kaggle.com/code/sthephannysantos/efficientnetv2-grad-cam-oral-histopathology).

---

<br>

## Sumário

- [Descrição dos Datasets](#-descrição-dos-datasets)
- [Arquiteturas Utilizadas](#-arquiteturas-utilizadas)
- [Modos de Treinamento](#️-modos-de-treinamento)
- [Estratégia de Data Augmentation](#-estratégia-de-data-augmentation)
- [Desenho Experimental](#-desenho-experimental)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Links e Downloads (Releases)](#-links-e-downloads-releases)
- [Como Executar o Projeto](#-como-executar-o-projeto)
- [Guia de Execução Passo a Passo](#-guia-de-execução-passo-a-passo)
- [Análise e Interpretação dos Resultados](#-análise-e-interpretação-dos-resultados)
- [Limitações Conhecidas](#️-limitações-conhecidas)

---

<br>

## Descrição dos Datasets

### Dataset 1 — OralEpitheliumDB
Desenvolvido pela Universidade Federal de Uberlândia (UFU). Focado em Displasia Epitelial Oral (OED), com imagens microscópicas de tecido bucal com potencial de transformação maligna. As imagens foram coletadas de 30 línguas de camundongo, com núcleos celulares marcados manualmente por especialista e rótulos avaliados e validados por patologista.

- **Total de imagens:** 456 (sendo 228 utilizadas após filtragem)
- **Tarefa:** Classificação binária
- **Subconjunto utilizado:** `Original ROI Images`
- **Fonte:** [LIPAI-Org/OralEpitheliumDB_Dataset](https://github.com/LIPAI-Org/OralEpitheliumDB_Dataset)

**Classes e divisão dos splits:**

| Split | healthy | severe | Total |
| :--- | :---: | :---: | :---: |
| Treino | 72 | 72 | 144 |
| Validação | 19 | 19 | 38 |
| Teste | 23 | 23 | 46 |
| **Total** | **114** | **114** | **228** |

<br>

### Dataset 2 — NDB-UFES
Desenvolvido pela Universidade Federal do Espírito Santo (UFES). Contém imagens histopatológicas de biópsias de pacientes atendidos pelo projeto de Diagnóstico Oral (NDB) entre 2010 e 2021. Além das imagens, o dataset inclui dados sociodemográficos (gênero, idade, cor da pele) e clínicos (tabagismo, consumo de álcool, exposição solar) — utilizados apenas como contexto, não no treinamento.

- **Total de imagens:** 237
- **Tarefa:** Classificação multiclasse (3 classes)
- **Fonte:** [Mendeley Data - NDB-UFES](https://data.mendeley.com/datasets/bbmmm4wgr8/4)

**Classes e divisão dos splits:**

| Split | Leukoplakia w/ dysplasia | Leukoplakia w/o dysplasia | OSCC | Total |
| :--- | :---: | :---: | :---: | :---: |
| Treino | 59 | 39 | 61 | 159 |
| Validação | 15 | 9 | 15 | 39 |
| Teste | 15 | 9 | 15 | 39 |
| **Total** | **89** | **57** | **91** | **237** |

> [!NOTE]
> Os splits de ambos os datasets são fixos e compartilhados para permitir a comparação direta dos resultados entre diferentes execuções.

---

<br>

## Arquiteturas Utilizadas

Ambas as arquiteturas são carregadas via biblioteca [TIMM (Pytorch Image Models)](https://github.com/huggingface/pytorch-image-models):

- [EfficientNetV2-B0](https://huggingface.co/timm/tf_efficientnetv2_b0.in1k) (`tf_efficientnetv2_b0.in1k`)
- [EfficientNetV2-B1](https://huggingface.co/timm/tf_efficientnetv2_b1.in1k) (`tf_efficientnetv2_b1.in1k`)

Cada modelo é composto por um **backbone** (responsável pela extração automática de características visuais) e uma **camada classificadora fully connected** (responsável pela predição das classes). Os pesos pré-treinados na ImageNet são baixados automaticamente pelo `timm` na primeira execução.

---

<br>

## Modos de Treinamento

Para cada arquitetura e dataset, três abordagens de transferência de aprendizado foram avaliadas:

| Modo (código) | Nome | Descrição |
| :---: | :--- | :--- |
| `scratch` | From Scratch | Todos os pesos inicializados aleatoriamente; treinamento do zero absoluto. |
| `feature_extraction` | Pré-treinado + Backbone Congelado | Pesos pré-treinados carregados; backbone congelado; apenas a camada classificadora final é treinada. |
| `fine_tuning` | Pré-treinado + Fine-tuning Completo | Pesos pré-treinados carregados; todas as camadas (backbone e classificador) são otimizadas. |

---

<br>

## Estratégia de Data Augmentation

As transformações de aumento de dados são aplicadas **exclusivamente no conjunto de treino**. Validação e teste utilizam apenas o redimensionamento padrão e a normalização.

- **Sem Augmentation:** `CenterCrop(224)` + Normalização ImageNet.
- **Com Augmentation:**
  - `RandomCrop(224)`
  - `RandomHorizontalFlip` (probabilidade de 50%)
  - `RandomRotation` sorteando estritamente entre 0°, 90°, 180° ou 270° (25% de chance cada)
  - Normalização padrão ImageNet (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)

---

<br>

## Desenho Experimental

```text
2 arquiteturas × 3 modos × 2 condições de augmentation × 2 datasets × 3 seeds = 72 execuções
```

- **Seeds utilizadas:** `42`, `123`, `2025`

Para mitigar o efeito da aleatoriedade, cada configuração é executada 3 vezes (uma por seed). As métricas finais reportadas consistem na **média** e no **desvio padrão** obtidos nas repetições.

**Protocolo de Hiperparâmetros:**

| Parâmetro | Valor |
| --- | --- |
| Épocas Máximas | 50 |
| Otimizador | Adam |
| Taxa de Aprendizado (LR) | **1e-5** (fixo para ambos os datasets) |
| Tamanho do Batch | 32 |
| Função de Perda | Cross Entropy (sem pesos) |
| Early Stopping | **Patience = 30** (baseado na acurácia de validação) |
| Critério de Seleção | Maior Acurácia de Validação |
| Mixed Precision | Sim (GradScaler do PyTorch) |

> [!IMPORTANT]
> Foi utilizando GPU T4 x2 no Kaggle.

---

<br>

## Estrutura do Repositório

O código fonte e os recursos visuais estão organizados da seguinte forma:

```text
├── img/
│   ├── help/             # Imagens do passo a passo para execução
│   └── analitic/         # Gráficos e imagens para interpretação dos resultados
├── notebooks/
│   └── efficientnetv2-grad-cam-oral-histopathology.ipynb # Versão para Kaggle
└── README.md
```

> [!WARNING]
> **Nota sobre arquivos pesados:** Arquivos de dados e resultados volumosos não são armazenados diretamente no histórico do Git. Eles estão hospedados de forma segura na aba de **[Releases](https://github.com/LIPAI-Org/Oral-Classification-EfficientNetV2/releases/tag/v1.0.0)** deste repositório.

---

<br>

## Links e Downloads (Releases)

Caso opte por rodar o projeto localmente ou inspecionar as saídas brutas, faça o download direto pelos assets da Release:

- **[datasets.zip](https://github.com/LIPAI-Org/Oral-Classification-EfficientNetV2/releases/download/v1.0.0/datasets.zip):** O conjunto de dados unificado utilizado em ambas as plataformas.
- **[resultados_colab.zip](https://www.kaggle.com/code/sthephannysantos/efficientnetv2-grad-cam-oral-histopathology):** Arquivo notebook hospedado e atualizado no Kaggle.

---

<br>

## Como Executar o Projeto

Escolha uma das plataformas abaixo para reproduzir ou estender a análise:

### Ambiente Kaggle (Recomendado)

A execução no Kaggle é direta, pois o ambiente e o dataset já estão integrados e hospedados na própria plataforma:

- 🔗 **Notebook Completo:** [EfficientNetV2 + Grad-CAM Oral Histopathology](https://www.kaggle.com/code/sthephannysantos/efficientnetv2-grad-cam-oral-histopathology)
- 🔗 **Dataset Original:** [Oral Dataset no Kaggle](https://www.kaggle.com/datasets/sthephannysantos/oral-dataset)

---

<br>

## Guia de Execução Passo a Passo

Siga o guia visual caso tenha dúvidas sobre como configurar o ambiente em cada site:

### Ambiente Kaggle (3 Passos)

| Passo | Descrição | Visualização |
| :---: | :--- | :---: |
| **1** | Acesse o notebook pelo link e clique em "Copy & Edit" | ![Passo 1 Kaggle](img/help/PASSO%201%20KAGGLE.png) |
| **2** | Verifique se o dataset `oral-dataset` está adicionado como input | ![Passo 2 Kaggle](img/help/PASSO%202%20KAGGLE.png) |
| **3** | Execute todas as células sequencialmente | ![Passo 3 Kaggle](img/help/PASSO%203%20KAGGLE.png) |

---

<br>

## Análise e Interpretação dos Resultados

Nesta seção são consolidadas as principais métricas obtidas e a interpretação visual gerada pelo Grad-CAM. As imagens analíticas completas podem ser consultadas na pasta `img/analitic/`.

### Matriz de Confusão e Curvas de Aprendizado

Os gráficos de perda (loss) e acurácia demonstram o comportamento do modelo ao longo das 50 épocas. A convergência e a capacidade de generalização variaram sensivelmente entre os modos `fine_tuning` e `scratch`.

### Visualização de Regiões de Interesse (Grad-CAM)

O uso do Grad-CAM permitiu mapear quais estruturas celulares e texturas teciduais foram determinantes para que a EfficientNetV2 categorizasse as patologias orais. Áreas hipercoradas e regiões com alterações morfológicas severas no epitélio tendem a registrar maior peso nos mapas de calor.

---

<br>

## Limitações Conhecidas

- O dataset **NDB-UFES** apresenta um desbalanceamento acentuado de classes (especialmente na classe `Leukoplakia without dysplasia`), o que pode impactar negativamente a métrica de F1-score macro.
- O limite estrito de 50 épocas pode ser insuficiente para a convergência plena nos testes realizados sob o modo `scratch`, principalmente no cenário multiclasse.
- Risco iminente de sobreajuste (overfitting) nos cenários de ajuste fino (`fine_tuning`) aplicados ao **OralEpitheliumDB**, decorrente do tamanho amostral reduzido do dataset (228 imagens totais compartilhadas).
- A taxa de aprendizado fixa em 1e-5 para ambos os datasets pode não ser ótima para todas as configurações; ajustes finos por modo de treinamento poderiam melhorar os resultados.
