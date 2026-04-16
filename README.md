# PD2 — Operacionalização de um Modelo de Risco de Crédito

Aplicação em Streamlit para estimar a probabilidade de inadimplência e classificar o risco com um modelo campeão treinado em scikit-learn.

## Visão geral

Este repositório representa o PD2 da disciplina e continua o trabalho iniciado no PD1, no qual o problema de risco de crédito foi explorado em notebook com foco em experimentação e comparação de modelos supervisionados. Aqui, o mesmo problema foi reorganizado em uma estrutura mais próxima de engenharia de machine learning, com separação de responsabilidades, rastreabilidade de experimentos e uma interface de inferência para uso prático.

O objetivo do projeto é estimar a probabilidade de inadimplência a partir de atributos financeiros e comportamentais, exibindo no app uma previsão de risco e a respectiva classificação entre menor e maior risco.

A avaliação do modelo considera métricas de classificação adequadas para dados desbalanceados e também diferentes thresholds de decisão. Essa análise complementa o PD1 porque mostra, de forma mais aplicada, como a regra de corte altera o equilíbrio entre precision e recall e, portanto, o perfil de risco da decisão final. Em scikit-learn, o corte padrão de classificação binária é 0.5, mas esse limiar pode ser ajustado conforme o objetivo do problema. 


## Demonstração

- Probabilidade estimada de inadimplência.
- Classificação binária de risco.
- Threshold ajustável no app.
- Interface web com Streamlit.

## Propósito do projeto

Este repositório organiza a segunda etapa do trabalho de risco de crédito em uma estrutura mais próxima do que se espera de um projeto de machine learning em operação. A ideia foi sair do formato centrado em notebook e reorganizar o fluxo em módulos, configuração centralizada, experimentos rastreáveis e interface de inferência.

O problema permanece o mesmo: prever inadimplência severa (`SeriousDlqin2yrs`) no dataset **Give Me Some Credit** do Kaggle, com cerca de 150 mil registros. A diferença está na forma de trabalhar o ciclo de vida do modelo: dados, treino, avaliação, persistência e uso em aplicação.

## O que mudou nesta etapa

Nesta versão, o projeto passou a ser dividido em partes claras:

- leitura e tratamento de dados.
- construção de features.
- treino do pipeline.
- avaliação offline do modelo.
- versionamento de artefatos.
- inferência via Streamlit.
- parâmetros organizados em YAML.


## Estrutura do repositório

```text
pd_operacionalizacao_modelo_mlops/
├── app/
│   └── app.py
├── configs/
│   └── experiment.yaml
├── data/
│   └── raw/
├── models/
├── scripts/
│   ├── evaluate_model.py
│   └── run_experiments.py
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── features.py
│   ├── inference.py
│   └── training.py
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── README.md
```

## Organização do código

- `src/`: contém a lógica reutilizável do projeto.
- `scripts/`: concentra as execuções principais de treino e avaliação.
- `configs/`: guarda os parâmetros do experimento.
- `app/`: interface Streamlit para simulação de inferência.
- `data/raw/`: local esperado para o dataset original.
- `models/`: onde ficam os artefatos gerados pelo fluxo.
- `tests/`: testes do pipeline e validações básicas.

## Dataset

O projeto usa o dataset **Give Me Some Credit**, com a coluna alvo `SeriousDlqin2yrs`.

O arquivo esperado deve estar em:

```text
data/raw/cs-training.csv
```

Se o dataset vier com a coluna `Unnamed: 0`, ela será removida no carregamento.

## Configuração centralizada

Os principais parâmetros do projeto ficam em:

```text
configs/experiment.yaml
```

Esse arquivo concentra configurações como:
- caminho do dataset;
- target;
- divisão treino/teste;
- tipo de modelo;
- hiperparâmetros;
- nome do experimento no MLflow;
- thresholds usados na avaliação.

## Fluxo principal

O projeto foi desenhado para seguir esta sequência:

1. Carregar os dados.
2. Preparar as features.
3. Treinar o pipeline.
4. Avaliar o modelo em conjunto de teste.
5. Salvar o modelo campeão.
6. Rodar avaliação offline com thresholds.
7. Abrir a interface de inferência.

## Métricas e avaliação

A avaliação do modelo considera métricas clássicas para classificação binária, com atenção especial ao desbalanceamento da classe positiva.

As saídas esperadas incluem:
- accuracy;
- precision;
- recall;
- f1-score;
- roc_auc;
- matriz de confusão;
- análise por threshold;
- importâncias de variáveis ou coeficientes.

## Como executar

### 1. Criar ambiente virtual

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Instalar dependências

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 3. Colocar o dataset no local esperado

```bash
mkdir -p data/raw
cp /caminho/para/cs-training.csv data/raw/
```

### 4. Rodar o treino

```bash
python scripts/run_experiments.py
```

Esse comando deve gerar o modelo campeão e registrar a execução no MLflow.

### 5. Rodar a avaliação offline

```bash
python scripts/evaluate_model.py
```

Esse script gera os arquivos auxiliares de análise e permite comparar o desempenho do modelo em diferentes thresholds.

### 6. Abrir o MLflow

```bash
mlflow ui
```

Depois, acesse:

```text
http://127.0.0.1:5000
```

### 7. Abrir a interface Streamlit

```bash
streamlit run app/app.py
```

Se necessário:

```bash
python -m streamlit run app/app.py
```

## Saídas esperadas

Ao rodar o fluxo completo, o projeto deve produzir:

- modelo treinado salvo localmente;
- `run_id` do experimento campeão;
- métricas de avaliação;
- CSV com análise de thresholds;
- arquivo com importâncias ou coeficientes;
- aplicação Streamlit para simular predições.

## Arquivos principais

- `src/data_processing.py`: leitura e preparação da base.
- `src/features.py`: lógica de features.
- `src/training.py`: treino do pipeline.
- `src/evaluation.py`: cálculo de métricas.
- `src/inference.py`: entrada e predição.
- `src/config.py`: leitura do YAML.
- `scripts/run_experiments.py`: execução do treino.
- `scripts/evaluate_model.py`: avaliação offline.
- `app/app.py`: interface de uso do modelo.

## Observações de organização

A pasta `src/` deve ficar apenas com código reutilizável.  
A pasta `scripts/` deve conter só pontos de execução.  
A pasta `models/` armazena apenas artefatos gerados.  
A pasta `data/raw/` deve conter somente o dataset de entrada.

## Observações

O threshold pode ser ajustado para diferentes cenários de negócio.


## Tecnologias

- Python
- Pandas
- NumPy
- scikit-learn
- Streamlit
- joblib
- MLflow
- PyYAML