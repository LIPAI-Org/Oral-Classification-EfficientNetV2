# Avaliação de Curvas de Treinamento

**Escopo:** triagem sistemática dos 72 experimentos do projeto

---
<br>

## Por que isso importa

O objetivo de um modelo não é memorizar o conjunto de treino. É generalizar — acertar em dados que nunca viu. Essa distinção parece óbvia, mas durante a avaliação de dezenas de experimentos é fácil confundir "treino convergiu" com "modelo útil". As curvas de loss e acurácia são o instrumento diagnóstico central: elas mostram se o que o modelo aprendeu tem alguma chance de funcionar fora do laboratório.

Este documento organiza os critérios de avaliação, os padrões visuais a identificar, e a tabela de triagem para registrar o diagnóstico de cada experimento.


---
<br>

## O que estamos olhando

A cada época de treinamento, duas métricas são registradas duas vezes: uma sobre os dados de treino, outra sobre os dados de validação.

**Loss** mede o quão erradas estão as predições. Deve cair ao longo do tempo.  
**Acurácia** mede a proporção de acertos. Deve subir.

O que interessa não é o valor absoluto de nenhuma delas isoladamente — é o comportamento *conjunto* das duas curvas (treino vs. validação) ao longo das épocas. A distância entre elas é o **gap de generalização**: quanto menor e mais estável, melhor.

> Referências principais usadas nesta análise:
> - [D2L — Underfitting e Overfitting](https://pt.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html)
> - [Google ML Crash Course — Interpretando curvas de loss](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves)
> - [Google ML Crash Course — Overfitting](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting)


---
<br>

## Os quatro padrões

<br>

### A — Bom fit ✅

As curvas de treino e validação descem juntas e convergem. O gap entre elas é pequeno e estável ao longo das épocas. A acurácia de validação cresce e se sustenta.

| Sinal | Comportamento esperado |
| :--- | :--- |
| Loss treino | Cai de forma consistente |
| Loss validação | Acompanha o treino, paralela |
| Gap treino–val | Pequeno, sem tendência de crescimento |
| Acurácia val | Cresce e se aproxima da de treino |

**O que fazer:** selecionar para análise aprofundada.


---
<br>

### B — Overfitting ❌

O modelo decorou o conjunto de treino. A loss de treino continua caindo, mas a de validação para ou começa a subir — as curvas formam uma tesoura. Na acurácia, o efeito invertido: treino sobe, validação estagna ou degenera.

| Sinal | Comportamento esperado |
| :--- | :--- |
| Loss treino | Cai continuamente |
| Loss validação | Cai no início, depois sobe ou explode |
| Gap treino–val | Cresce ao longo das épocas |
| Acurácia val | Estagna ou cai após pico inicial |

**O que fazer:** descartar, ou investigar regularização se o overfitting for leve.


---
<br>

### C — Underfitting ❌

Aqui o gap é pequeno — mas isso não é bom sinal. Ambas as curvas ficam presas em patamares ruins, sem tendência clara de queda. O modelo não tem capacidade (ou tempo) suficiente para aprender os padrões.

| Sinal | Comportamento esperado |
| :--- | :--- |
| Loss treino | Permanece alta, cai pouco |
| Loss validação | Também permanece alta |
| Gap treino–val | Pequeno, mas ambas em nível ruim |
| Acurácia val | Baixa e sem melhora |

**O que fazer:** descartar. Gap pequeno não é virtude quando o nível absoluto é ruim.

---
<br>

### D — Instável ⚠️

As curvas oscilam muito, sem tendência clara. A validação pode ter um pico inicial bom que não se sustenta. O comportamento muda entre seeds. Pode ser taxa de aprendizado alta, batch pequeno ou inicialização desfavorável.

| Sinal | Comportamento esperado |
| :--- | :--- |
| Loss treino | Oscila, sem tendência clara |
| Loss validação | Idem, ou degenera após início promissor |
| Gap treino–val | Variável, inconsistente |
| Acurácia val | Picos seguidos de quedas bruscas |

**O que fazer:** revisar antes de descartar. Se houver tendência de queda perceptível sob o ruído, o experimento pode ser recuperável.


---
<br>

## Como avaliar cada imagem

**Passo 1 — Gráfico de loss (esquerdo)**

- A loss de treino cai ao longo das épocas?
- A loss de validação acompanha, ou diverge?
- O gap entre as duas é estável?

Se os três são sim, candidato ao Padrão A.

<br>

**Passo 2 — Gráfico de acurácia (direito)**

- A acurácia de validação cresce ou se sustenta?
- Ela não colapsa após um pico inicial?
- O nível absoluto é razoável para o problema?

<br>

**Passo 3 — Identificar o padrão dominante**

| O que você vê | Padrão |
| :--- | :---: |
| Ambas caem juntas, gap pequeno e estável | **A** |
| Treino cai, val sobe — gap crescente | **B** |
| Ambas ficam altas e planas | **C** |
| Curvas serrilhadas, picos que não se sustentam | **D** |

<br> 

**Passo 4 — Registrar na tabela de triagem.**


---
<br>

## Sinais específicos deste experimento

Vale ter em mente algumas particularidades do setup antes de avaliar:

- **Feature extraction com backbone congelado** significa que apenas as camadas densas finais são treinadas. Essas camadas são mais sensíveis à inicialização — o que explica a diferença de comportamento entre seeds.
- **Sem augmentation** (`aug=False`): o modelo não recebe variação artificial dos dados. Em datasets pequenos como o NDB-UFES, isso aumenta o risco de overfitting.
- **Acurácia baixa em geral** é esperada aqui. As features do EfficientNetV2 foram aprendidas no ImageNet (fotografias do cotidiano), não em imagens dermatológicas. A distância de domínio é real.
- **Picos isolados de validação nas primeiras épocas** são ruído, não sinal de aprendizado.
- **Val loss subindo enquanto train loss oscila** (como observado no `seed42`) indica que o classificador não convergiu — as representações aprendidas são instáveis.


---
<br>

## Tabela de triagem

(AINDA FALTA ESCOLHER learning_curves PARA POR NO REPO E ANALISAR AQUI)

| # | Arquivo | Padrão | Loss treino final | Loss val final | Acurácia val máx | Observação | Decisão |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- | :--- |
| 1 | | | | | | | |
| 2 | | | | | | | |
| 3 | | | | | | | |
| 4 | | | | | | | |
| 5 | | | | | | | |
| 6 | | | | | | | |
| 7 | | | | | | | |
| 8 | | | | | | | |
| 9 | | | | | | | |
| 10 | | | | | | | |


---
<br>

## Critérios de seleção final

Depois de preencher a tabela, a ordem de prioridade é:

1. **Padrão A** — convergência real, gap estável, acurácia de val sustentada.
2. **Padrão D com tendência perceptível** — instável, mas com direção de queda clara sob o ruído.
3. **Padrão B leve** — overfitting pequeno, val loss não explodiu, acurácia ainda razoável.
4. **Padrão B severo e Padrão C** — descartar, ou usar apenas como baseline negativo para comparação.

Entre os experimentos selecionados, desempatar por:

- **Maior acurácia de validação sustentada** — não o pico isolado, mas o valor que se mantém nas últimas épocas.
- **Menor gap treino–val no final** — sinal de que o modelo ainda generaliza.
- **Menor variação nas últimas épocas** — curvas mais suaves indicam convergência mais confiável.


---
<br>

## Referências

| Fonte | Conceito | Link |
| :--- | :--- | :--- |
| Dive into Deep Learning (PT) | Underfitting, overfitting, seleção de modelo, gap de generalização | [Acessar](https://pt.d2l.ai/chapter_multilayer-perceptrons/underfit-overfit.html) |
| Google ML Crash Course | Leitura e interpretação de curvas de loss | [Acessar](https://developers.google.com/machine-learning/crash-course/overfitting/interpreting-loss-curves) |
| Google ML Crash Course | Definição de overfitting e generalização | [Acessar](https://developers.google.com/machine-learning/crash-course/overfitting/overfitting) |
