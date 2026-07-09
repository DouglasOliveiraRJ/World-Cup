# World-Cup

# 🚀 Preditor Inteligente e Simulador de Copa do Mundo 2026

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/SQL_Server-CC2927?style=for-the-badge&logo=microsoft-sql-server&logoColor=white" alt="SQL Server">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/SQLAlchemy-D71F27?style=for-the-badge&logo=redhat&logoColor=white" alt="SQLAlchemy">
  <img src="https://img.shields.io/badge/Scikit_Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy">
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge" alt="Seaborn">
</p>

## 📃 Visão Geral do Projeto

Este repositório contém o meu primeiro grande projeto prático de dados de ponta a ponta. O objetivo principal foi sair das bases de dados prontas e "perfeitas" da internet e enfrentar o desafio de estruturar, limpar, analisar e aplicar modelos de Machine Learning em dados reais do futebol mundial para simular a Copa do Mundo de 2026.

Como estou iniciando na área de dados, construí este projeto para consolidar meus fundamentos em SQL, Python e conceitos iniciais de Data Science. Em vez de apenas criar um modelo de Inteligência Artificial e aceitar o resultado às cegas, busquei entender os erros do algoritmo, aplicar inteligência de negócio através de Engenharia de Features e criar um motor estatístico (Simulação de Monte Carlo) que calcula probabilidades reais de classificação em vez de um resultado estático.

---

## 🚧 Arquitetura e Fluxo do Pipeline

O projeto foi estruturado seguindo uma esteira lógica de processamento de dados, garantindo o desacoplamento entre o armazenamento e a parte analítica:

```text
       [ Banco MySQL ]                      [ Integração ]               [ Sandbox Científica ]              [ Motor Estatístico ]
       Tabelas Locais              --->     SQLAlchemy / Pandas   --->   Análise & ML (Random Forest)  --->  Simulação Monte Carlo
(cadastro, historico, estrutura)           (Migração para VS Code)       (Baseline 60.31% -> 61.62%)         (10.000 rodadas via Proba)

```


---

## 🛠️ Detalhamento Passo a Passo

### 1. Modelagem de Dados e Integração
Em vez de utilizar arquivos CSV soltos, estruturei os dados brutos dentro do banco de dados relacional SQL Server 2022. Modelei e criei três tabelas essenciais: cadastro (atributos das seleções como valor de mercado e ranking da FIFA), historico (partidas antigas para treino do modelo) e estrutura (com a divisão oficial dos grupos e chaves da Copa de 2026). A integração e migração dos dados para o Python foram feitas utilizando SQLAlchemy e Pandas dentro do VS Code.

```python
# Tirando a notação científica
pd.set_option('display.float_format', lambda x: '%.0f' % x)

# Conexão com o banco
engine = create_engine('mysql+pymysql://root:********@localhost:3306/copa_do_mundo')

df_cadastro = pd.read_sql("SELECT * FROM cadastro", con=engine)
df_historico = pd.read_sql("SELECT * FROM historico", con=engine)
df_estrutura = pd.read_sql("SELECT * FROM estrutura", con=engine)

```

### 2. Análise Exploratória de Dados (1_analise_exploratoria.py)
Utilizando pd.crosstab e Matplotlib, investiguei o que realmente ganha jogo no futebol. Descobri um insight crítico de negócio: o Fator Casa é avassalador. Mesmo quando o time visitante é financeiramente mais valioso, o mandante consegue pontuar (vencer ou empatar) em 79% das vezes. Como a Copa do Mundo é jogada em campo neutro, entendi que precisaria isolar essa variável para não enviesar o modelo final.

```python
colunas_para_teste = [
    'vitoria',
    'valor_mercado_elenco',
    'rating_ea_fc_geral',
    'idade_media_elenco',
    'percentual_jogadores_elite',
    'ranking_fifa_pontos'
]

# Calculando a matriz de correlação
matriz_corr = df_analise[colunas_para_teste].corr()

# plotando o gráfico
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='RdYlGn', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlação das variantes com a Vitória')
plt.tight_layout()
plt.show()

```

### 3. O Primeiro Modelo Preditivo
Treinei um modelo clássico de Random Forest Classifier utilizando os atributos brutos das seleções mandantes e visitantes (como pontos no ranking FIFA e idade média).
- Resultado Baseline: Conseguimos uma acurácia geral inicial de 60.31%.

- O Diagnóstico Técnico: Analisando o relatório de classificação (classification_report), percebi que o modelo tinha um Recall excelente para vitórias (0.83), mas um desempenho péssimo para prever empates (apenas 0.08). O algoritmo com dados brutos simplesmente não conseguia identificar quando um jogo seria equilibrado na fase de grupos.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

features = [
    "valor_mercado_mandante",
    "valor_mercado_visitante",
    "ranking_fifa_pontos_mandante",
    "rating_ea_fc_geral_mandante",
    "idade_media_elenco_mandante",
    "percentual_jogadores_elite_mandante",
    "ranking_fifa_pontos_visitante",
    "rating_ea_fc_geral_visitante",
    "idade_media_elenco_visitante",
    "percentual_jogadores_elite_visitante",
]

X = df_modelo[features]
y = df_modelo['target']

# Separando os dados em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Treinando o modelo
modelo_copa = RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42
)
modelo_copa.fit(X_train, y_train)

# fazendo a previsão no grupo teste
y_pred = modelo_copa.predict(X_test)

# Acurácia geral do modelo
f"{accuracy_score(y_test, y_pred):.2%}"

# mostrar um relatório de desempenho
print(classification_report(y_test, y_pred, target_names=["Derrota", "Empate", "Vitória"]))

```
### 4. Engenharia de Features e Evolução
Para corrigir o problema dos empates e o viés de mando de campo, decidi criar novas variáveis que ajudassem o modelo a entender a disparidade real entre as equipes e simulassem o ambiente de Copa do Mundo. Criei as variáveis Delta, que calculam diretamente a diferença matemática entre os times em cada confronto, e isolei o mando definindo a flag fixada de campo neutro.

```python
# Criando as variáveis de confronto direto
df_modelo['delta_ranking_fifa'] = df_modelo['ranking_fifa_pontos_mandante'] - df_modelo['ranking_fifa_pontos_visitante']
df_modelo['delta_rating_ea'] = df_modelo['rating_ea_fc_geral_mandante'] - df_modelo['rating_ea_fc_geral_visitante']
df_modelo['delta_idade'] = df_modelo['idade_media_elenco_mandante'] - df_modelo['idade_media_elenco_visitante']
df_modelo['delta_elite'] = df_modelo['percentual_jogadores_elite_mandante'] - df_modelo['percentual_jogadores_elite_visitante']
df_modelo['delta_valor_mercado'] = df_modelo['valor_mercado_mandante'] - df_modelo['valor_mercado_visitante']

# Criando o fator campo neutro
df_modelo['campo_neutro'] = df_modelo['fator_campo'].map({'Neutro': 1, 'Casa': 0})
```
- Resultado Pós-Features: Com a inteligência dos Deltas e o ajuste para campo neutro, mitigamos o viés dos dados brutos e a acurácia geral do modelo saltou de 60.31% para 61.62%, com uma melhora expressiva na detecção de cenários equilibrados.


### 5. Motor de Simulação Estatística
Futebol não é uma ciência exata. Para capturar a imprevisibilidade do esporte, mudei a estratégia. Em vez de fazer o modelo dar uma resposta seca de quem ganha, utilizei o método predict_proba para extrair as probabilidades de cada resultado. Construí um loop de Monte Carlo que roda 10.000 simulações completas de todas as partidas de todos os grupos mapeados na tabela estrutura, gerando um sorteio ponderado.

```python
# Prevendo as probabilidades do modelo
probabilidade = modelo.predict_proba(dados_confronto)[0]

# Retorna o sorteio ponderado baseado nas probabilidades (0 = Derrota, 1 = Empate, 2 = Vitória)
return np.random.choice([0, 1, 2], p=probabilidade)
```

---

## 📈 Resultados e Outputs Obtidos
Após processar as 10.000 rodadas completas do motor de Monte Carlo (simulando desde os grupos até a grande final), o script consolida a taxa probabilística de sucesso de cada país. 
Abaixo está o formato do output gerado pelo modelo mostrando todas as 48 seleções, mas destacando as **Top 10 seleções com as maiores probabilidades de levantar a taça** da Copa do Mundo de 2026:

| Posição | Seleção | Grupos (%) | 16 avos (%) | Oitavas (%) | Quartas (%) | Semifinal (%) | Campeão (%)|
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | Argentina| 98.5% | 66.0% | 43.4% | 28.4% | 18.3% | 11.2% |
| 2 | Espanha | 93.7% | 61.0% | 38.9% | 24.3% | 14.9% | 9.4% |
| 3 | Portugal | 95.0% | 62.8% | 40.5% | 25.0% | 15.0% | 8.5% |
| 4 | Inglaterra | 96.7% | 61.9% | 37.7% | 22.4% | 13.8% | 7.8% |
| 5 | França | 96.1% | 59.5% | 35.7% | 20.9% | 12.0% | 6.7% |
| 6 | Brasil | 94.0% | 60.1% | 36.8% | 22.0% | 12.3% | 6.3% |
| 7 | Alemanha | 97.3% | 59.7% | 36.1% | 20.4% | 11.3% | 6.3% |
| 8 | Holanda | 94.3% | 55.4% | 30.6% | 17.0% | 8.7% | 4.2% |
| 9 | Uruguai | 93.2% | 51.5% | 27.8% | 14.5% | 7.2% | 3.4% |
| 10 | Bélgica | 83.8% | 47.3% | 25.9% | 13.4% | 6.8% | 3.3% |

---

## 🧠 Lições Aprendidas
Este projeto foi um divisor de águas nos meus estudos práticos de tecnologia. As principais lições que levo daqui são:

- Entendimento do Pipeline: Aprendi que Data Science não é só rodar um modelo de IA. A maior parte do trabalho está em pensar na estrutura do banco de dados, tratar as tabelas e entender o problema de negócio (como o impacto do fator campo e a dificuldade em cravar empates).

- Diagnóstico sobre Otimização Cega: Descobri a importância de ler um relatório de métricas técnico. Se eu tivesse olhado apenas a acurácia inicial de 60.31%, acharia que o modelo estava razoável, mas olhar o recall de 0.08 nos empates me forçou a estudar Engenharia de Features para corrigir um problema real e evoluir o modelo para 61.62%.

- Orquestração Lógica: Desenvolver funções e loops complexos (como a lógica de tabelas de pontos dentro de Monte Carlo) aprimorou drasticamente minha lógica de programação pura com dicionários e matrizes em Python.










