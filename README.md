# Predição da Duração de Viagens de Táxi em NYC - Projeto MLOps

## Visão Geral

Este projeto demonstra um fluxo de trabalho completo de Machine Learning para prever a duração de viagens de táxi na cidade de Nova York. Inclui processamento de dados, treinamento de modelos, avaliação e implantação usando práticas modernas de MLOps.

**Observação:** Devido à limitação de tamanho dos arquivos, os dados podem ser baixados em:

*   [https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)

## Estrutura do Projeto

*   `data/`: Contém os dados brutos (arquivos Parquet).
*   `models/`: Armazena modelos treinados (arquivos pickle).
*   `training/`: Contém o código de treinamento (`duration-prediction.ipynb`).
*   `mlruns/`: Diretório de rastreamento do MLflow (armazena execuções de experimentos, parâmetros, métricas e artefatos).
*   `README.md`: Este arquivo.

## Dados

O conjunto de dados usado neste projeto são os Registros de Viagens de Táxi Verde da Comissão de Táxis e Limusines de NYC. Os dados incluem datas/horas de embarque e desembarque, locais, distâncias de viagem e outras informações relevantes.

*   `green_tripdata_2024-01.parquet`: Dados de treinamento (janeiro de 2024).
*   `green_tripdata_2024-02.parquet`: Dados de validação (fevereiro de 2024).

## Fluxo de Trabalho

1.  **Ingestão e Pré-processamento de Dados:**
    *   Os dados são lidos de arquivos Parquet usando `pandas`.
    *   A duração da viagem é calculada a partir dos horários de embarque e desembarque.
    *   Valores discrepantes na duração da viagem são filtrados (1 a 60 minutos).
    *   Recursos categóricos (`PULocationID`, `DOLocationID`) são convertidos em strings.
    *   Um recurso combinado `PU_DO` é criado concatenando os IDs de local de embarque e desembarque.
2.  **Engenharia de Recursos:**
    *   `DictVectorizer` de `scikit-learn` é usado para converter recursos categóricos em recursos numéricos usando codificação one-hot.
3.  **Treinamento do Modelo:**
    *   **Regressão Linear:** Um modelo de `Regressão Linear` simples é treinado como linha de base.
    *   **Regressão Lasso:** Um modelo `Lasso` é treinado com regularização L1, e os resultados são rastreados usando MLflow.
    *   **XGBoost:** Um modelo XGBoost é treinado com otimização de hiperparâmetros usando Hyperopt.
4.  **Otimização de Hiperparâmetros:**
    *   Hyperopt é usado para pesquisar os melhores hiperparâmetros para o modelo XGBoost.
    *   A função `objective` treina e avalia o modelo XGBoost para um determinado conjunto de hiperparâmetros.
    *   O espaço de busca inclui parâmetros como `max_depth`, `learning_rate`, `reg_alpha`, `reg_lambda` e `min_child_weight`.
    *   O algoritmo TPE (Tree-structured Parzen Estimator) é usado para guiar a pesquisa.
5.  **Rastreamento de Experimentos:**
    *   MLflow é usado para rastrear todos os experimentos, incluindo parâmetros, métricas (RMSE) e artefatos (modelos treinados).
6.  **Persistência do Modelo:**
    *   Modelos treinados e objetos `DictVectorizer` são salvos usando `pickle`.

## Visão Geral do Código

*   **`training/duration-prediction.ipynb`:** Jupyter Notebook contendo o pipeline de treinamento completo.

    *   **Dependências:** `pandas`, `scikit-learn`, `mlflow`, `xgboost`, `hyperopt`.

## MLflow

MLflow é usado para rastreamento de experimentos. Para visualizar a interface do usuário do MLflow, execute:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db