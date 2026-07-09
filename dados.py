# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

# Tirando a notação científica
pd.set_option('display.float_format', lambda x: '%.0f' % x)

# %%
# Conexão com o banco
engine = create_engine('mysql+pymysql://root:dodo2002@localhost:3306/copa_do_mundo')

# %%
df_cadastro = pd.read_sql("SELECT * FROM cadastro", con=engine)
df_historico = pd.read_sql("SELECT * FROM historico", con=engine)
df_estrutura = pd.read_sql("SELECT * FROM estrutura", con=engine)

# %%
df_cadastro.head()

# %%
df_historico.head()

# %%
df_estrutura.head()

# %%
# Definir o id_selecao como índice do cadastro
df_cadastro = df_cadastro.set_index('id_selecao')

# Definir o id_partida como índice do histórico de jogos
df_historico = df_historico.set_index('id_partida')

# Definir o id_estrutura como índice da estrutura dos grupos
df_estrutura = df_estrutura.set_index('id_estrutura')

# %%
# Cruzando a coluna fator_campo com resultado e ver a porcentagem de V, E, D como mandante e visitante
tabela_fator_casa = pd.crosstab(
    df_historico['fator_campo'], 
    df_historico['resultado'], 
    normalize='index') * 100

# %%
tabela_fator_casa

# %%
# Unificar 2 tabelas para trazer o valor de mercado do time mandante
df_mercado = pd.merge(
    df_historico,
    df_cadastro[['valor_mercado_elenco']],
    left_on='id_selecao_mandante',
    right_index=True,
    how='left'
).rename(columns={'valor_mercado_elenco': 'valor_mercado_mandante'})

# %%
# Agora trazer o valor de mercado do time visitante
df_mercado = pd.merge(
    df_mercado,
    df_cadastro[['valor_mercado_elenco']],
    left_on='id_selecao_visitante',
    right_index=True,
    how='left'
).rename(columns={'valor_mercado_elenco': 'valor_mercado_visitante'})

# %%
# Criar uma coluna para mostrar quem é o time mais valioso
df_mercado['maior_valor'] = np.where(
    df_mercado['valor_mercado_mandante'] > df_mercado['valor_mercado_visitante'],
    'Mandante mais caro',
    'Visitante mais caro'
)

# %%
df_mercado

# %%
# Cruzando as colunas para receber a % dos resultados dos times mais caros
tabela_valor_mercado = pd.crosstab(
    df_mercado['maior_valor'], 
    df_mercado['resultado'], 
    normalize='index') * 100

# %%
tabela_valor_mercado

# Foi percebido que mesmo time visitante sendo mais caro, o time mandante 
# arranca uma vitória ou empate em 79% das vezes devido ao fator casa. 
# com isso vamos filtrar apenas campos neutros visto que o objetivo final é prever a copa do mundo.

# %%
# Filtro para campo neutro
df_mercado_neutro = df_mercado[df_mercado['fator_campo'] == 'Neutro']

# %%
tabela_mercado_neutro = pd.crosstab(
    df_mercado_neutro['maior_valor'],
    df_mercado_neutro['resultado'],
    normalize='index'
    ) * 100

# %%
tabela_mercado_neutro

# %%
# Trazendo as colunas de outro DF para poder fazer a correlação
df_analise = pd.merge(
    df_mercado,
    df_cadastro[
        [
            'ranking_fifa_pontos',
            'idade_media_elenco',
            'percentual_jogadores_elite',
            'valor_mercado_elenco',
            'rating_ea_fc_geral'
        ]
    ],
    left_on='id_selecao_mandante',
    right_index=True,
    how='left',
)

# %%
# Criando uma variável para utilizar no grafico de correlação 
df_analise['vitoria'] = (df_analise['resultado']== 'V').astype(int)

# %%
colunas_para_teste = [
    'vitoria',
    'valor_mercado_elenco',
    'rating_ea_fc_geral',
    'idade_media_elenco',
    'percentual_jogadores_elite',
    'ranking_fifa_pontos'
]

# %%
# Calculando a matriz de correlação
matriz_corr = df_analise[colunas_para_teste].corr()

# %%
# plotando o gráfico
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap='RdYlGn', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlação das variantes com a Vitória')
plt.tight_layout()
plt.show()

# %%
# Criar um DF modelo para usar no Preditor de 90 minutos que vamos utilizar para prever os jogos da fase de grupos

# Selecionando as colunas para o modelo
colunas_atributos = [
    'ranking_fifa_pontos',
    'rating_ea_fc_geral',
    'idade_media_elenco',
    'percentual_jogadores_elite',
]

# %%
# Trazendo os atributos do mandante
df_modelo = pd.merge(
    df_mercado,
    df_cadastro[colunas_atributos],
    left_on='id_selecao_mandante',
    right_index=True,
    how='left',
)

# %%
# Renomenando as colunas que foram pegas no merge para identificar que são do mandante
df_modelo = df_modelo.rename(
    columns={col: f'{col}_mandante' for col in colunas_atributos}
)

# %%
# Trazer os atributos do visitante

df_modelo = pd.merge(
    df_modelo,
    df_cadastro[colunas_atributos],
    left_on='id_selecao_visitante',
    right_index=True,
    how='left',
)

# %%
# Renomeando as colunas que foram pegas no merge para identificar que são do visitante
df_modelo = df_modelo.rename(
    columns={col: f'{col}_visitante' for col in colunas_atributos}
)

# %%
# Criando o target para a fase de grupos do modelo
# V = 2 (vitória mandante), E = 1 (empate), D = 0 (vitória visitante)
df_modelo['target'] = df_modelo['resultado'].map({'V': 2, 'E': 1, 'D': 0})

# %%
df_modelo.info()

# %%
# Treinando o modelo preditor (Random Forest Multiclasse)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

# %%
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

# %%
# Separando os dados em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %%
# Treinando o modelo
modelo_copa = RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42
)
modelo_copa.fit(X_train, y_train)

# %%
# fazendo a previsão no grupo teste
y_pred = modelo_copa.predict(X_test)


# %%
# Acurácia geral do modelo
f"{accuracy_score(y_test, y_pred):.2%}"

# %%
# mostrar um relatório de desempenho
print(classification_report(y_test, y_pred, target_names=["Derrota", "Empate", "Vitória"]))


# Batemos a marca dos 60.31% de acurácia geral o que é bom para um modelo preditivo com 3 classes.
# Mas, analisando o relatório, temos 2 questões. O modelo para prever a vitória está indo bem, teve um Recall de 0.83. 
# Isso significa que, de cada 100 jogos que o mandante realmente venceu, o modelo acertou 83 deles! 
# A precisão de 0.68 também é boa.

# A parte ruim é que, para prever o empate, foi de apenas 0.08. O modelo quase nunca consegue cravar um empate, 
# e isso, para o nosso objetivo final, que conta com a fase de grupos, é prejudicial. 
# Com isso, vamos usar o predict_proba, que nos diz a porcentagem de vitória, empate e derrota, 
# ao invés de dar somente o resultado que ele previu.


# %%
# vamos testar o fator campo neutro e a criação de variáveis deltas (Em vez de dar o Overall do Mandante e do Visitante separados, 
# se criarmos uma coluna dif_overall = overall_mandante - overall_visitante, o modelo passa a entender instantaneamente quem é o favorito 
# e o tamanho da disparidade. Com isso, veremos se vamos obter alguma melhora na acurácia.)


# Criando as variáveis de confronto direto
df_modelo['delta_ranking_fifa'] = df_modelo['ranking_fifa_pontos_mandante'] - df_modelo['ranking_fifa_pontos_visitante']
df_modelo['delta_rating_ea'] = df_modelo['rating_ea_fc_geral_mandante'] - df_modelo['rating_ea_fc_geral_visitante']
df_modelo['delta_idade'] = df_modelo['idade_media_elenco_mandante'] - df_modelo['idade_media_elenco_visitante']
df_modelo['delta_elite'] = df_modelo['percentual_jogadores_elite_mandante'] - df_modelo['percentual_jogadores_elite_visitante']
df_modelo['delta_valor_mercado'] = df_modelo['valor_mercado_mandante'] - df_modelo['valor_mercado_visitante']

# %%
# Criando o fator campo neutro
df_modelo['campo_neutro'] = df_modelo['fator_campo'].map({'Neutro': 1, 'Casa': 0})

# %%
features_novas = [
    'ranking_fifa_pontos_mandante', 'ranking_fifa_pontos_visitante',
    'rating_ea_fc_geral_mandante', 'rating_ea_fc_geral_visitante',
    'campo_neutro',
    'delta_ranking_fifa',
    'delta_rating_ea',
    'delta_idade',
    'delta_elite',
    'delta_valor_mercado'
]

X = df_modelo[features_novas]
y = df_modelo['target']

# %%
# Dividindo o novo modelo
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,  stratify=y
)

# %%
# Treinando o novo modelo
novo_modelo_copa = RandomForestClassifier(
    n_estimators=100, max_depth=8, random_state=42
)
novo_modelo_copa.fit(X_train, y_train)

# %%
# Fazendo a previsão
y_pred = novo_modelo_copa.predict(X_test)

# %%
# Acurácia geral do modelo
f"{accuracy_score(y_test, y_pred):.2%}"

# %%
# Mostrando o relatório de desempenho
print(classification_report(y_test, y_pred, target_names=["Derrota", "Empate", "Vitória"]))

# %%
# Pegando a ultima aparição de cada seleção para ter os dados mais recentes delas
df_cadastro = (
    df_modelo.sort_values("data_partida")
    .groupby("id_selecao_mandante")
    .last()[
        [
            "ranking_fifa_pontos_mandante",
            "rating_ea_fc_geral_mandante",
            "idade_media_elenco_mandante",
            "percentual_jogadores_elite_mandante",
            "valor_mercado_mandante",
        ]
    ]
)

# %%
# Renomeando as colunas
df_cadastro.columns = [
    'ranking_fifa_pontos',
    'rating_ea_fc_geral',
    'idade_media_elenco',
    'percentual_jogadores_elite',
    'valor_mercado',
]


# %%
def simular_partida_mc(id_m, id_v, modelo, df_cad):
# Buscar os atributos de cada time
    atrib_m = df_cad.loc[id_m]
    atrib_v = df_cad.loc[id_v]

# Montando o dataframe de entrada
    dados_confronto = pd.DataFrame(
    [
        {
            'ranking_fifa_pontos_mandante': atrib_m['ranking_fifa_pontos'],
            'ranking_fifa_pontos_visitante': atrib_v[
                'ranking_fifa_pontos'
            ],
            'rating_ea_fc_geral_mandante': atrib_m['rating_ea_fc_geral'],
            'rating_ea_fc_geral_visitante': atrib_v['rating_ea_fc_geral'],
            'campo_neutro': 1,
            'delta_ranking_fifa': atrib_m['ranking_fifa_pontos'] - atrib_v['ranking_fifa_pontos'],
            'delta_rating_ea': atrib_m['rating_ea_fc_geral'] - atrib_v['rating_ea_fc_geral'],
            'delta_idade': atrib_m['idade_media_elenco'] - atrib_v['idade_media_elenco'],
            'delta_elite': atrib_m['percentual_jogadores_elite'] - atrib_v['percentual_jogadores_elite'],
            'delta_valor_mercado': atrib_m['valor_mercado'] - atrib_v['valor_mercado'],
        }
    ]
)
    # Ordena as colunas
    dados_confronto = dados_confronto[features_novas]

    # Prevendo as probabilidades do modelo
    probabilidade = modelo.predict_proba(dados_confronto)[0]

    # 5. Retorna o sorteio ponderado (0 = Derrota, 1 = Empate, 2 = Vitória)
    return np.random.choice([0, 1, 2], p=probabilidade)

# %%
# Vou usar o loop de Monte Carlo para rodar 10.000 simulações e no final receber a probabilidade de classificação

# Mapeamento dos grupos da copa
copa_2026_grupos = {
    "Grupo A": {"México": 25, "Coreia do Sul": 32, "República Tcheca": 14, "África do sul": 38},
    "Grupo B": {"Suíça": 16, "Canadá": 23, "Catar": 31, "Bósnia e Herzegovina": 5},
    "Grupo C": {"Escócia": 7, "Marrocos": 44, "Brasil": 1, "Haiti": 27},
    "Grupo D": {"Estados Unidos": 24, "Austrália": 30, "Turquia": 17, "Paraguai": 21},
    "Grupo E": {"Alemanha": 2,"Costa do Marfim": 41,"Equador": 20,"Curaçao": 26},
    "Grupo F": {"Suécia": 15,"Japão": 35,"Holanda": 12,"Tunísia": 46},
    "Grupo G": {"Nova Zelândia": 48,"Irã": 33,"Bélgica": 4,"Egito": 42},
    "Grupo H": {"Uruguai": 22,"Arábia Saudita": 29,"Espanha": 8,"Cabo Verde": 40},
    "Grupo I": {"França": 9,"Senegal": 45,"Iraque": 34,"Noruega": 11},
    "Grupo J": {"Argentina": 18,"Argélia": 39,"Áustria": 3,"Jordânia": 36},
    "Grupo K": {"Portugal": 13,"RD Congo": 47,"Uzbequistão": 37,"Colômbia": 19},
    "Grupo L": {"Inglaterra": 10,"Croácia": 6,"Gana": 43,"Panamá": 28},
}


# %%
# Exibição dos resultados

nome_para_id = {}
for grupo_nome, times_dict in copa_2026_grupos.items():
    for nome, id_time in times_dict.items():
        nome_para_id[nome] = id_time

N_SIMULACOES = 10000

# Dicionário para registrar o sucesso histórico de cada seleção 
historico_mata_mata = {
    time: {"R32 (Classif.)": 0, "R16 (Oitavas)": 0, "QF (Quartas)": 0, "SF (Semi)": 0, "F (Final)": 0, "Campeão": 0}
    for time in nome_para_id.keys()
}

print(f"🔮 Rodando {N_SIMULACOES} simulações completas da Copa (Grupos + Novo Mata-mata)...")

# EXECUÇÃO DO MULTIVERSO
for _ in range(N_SIMULACOES):
    
    # FASE DE GRUPOS
    primeiros_lugares = []
    segundos_lugares = []
    todos_terceiros = []

    for grupo_nome, times_dict in copa_2026_grupos.items():
        times_grupo = list(times_dict.keys())
        tabela_rodada = {time: 0 for time in times_grupo}

        for i in range(len(times_grupo)):
            for j in range(i + 1, len(times_grupo)):
                t1, t2 = times_grupo[i], times_grupo[j]
                
                # Executa o "motor" que está rodando 
                resultado = simular_partida_mc(nome_para_id[t1], nome_para_id[t2], novo_modelo_copa, df_cadastro)

                if resultado == 2:
                    tabela_rodada[t1] += 3
                elif resultado == 1:
                    tabela_rodada[t1] += 1
                    tabela_rodada[t2] += 1
                else:
                    tabela_rodada[t2] += 3

        ranking = sorted(tabela_rodada.items(), key=lambda x: x[1], reverse=True)
        
        primeiros_lugares.append(ranking[0]) # (Nome, Pontos)
        segundos_lugares.append(ranking[1])
        todos_terceiros.append(ranking[2])

    # Seleciona os 8 melhores terceiros colocados por pontos
    todos_terceiros_ordenados = sorted(todos_terceiros, key=lambda x: x[1], reverse=True)
    melhores_terceiros = todos_terceiros_ordenados[:8]

    # Monta o bolo dos 32 classificados para os 16 avos de final
    classificados_32 = [t[0] for t in primeiros_lugares] + [t[0] for t in segundos_lugares] + [t[0] for t in melhores_terceiros]
    
    for time in classificados_32:
        historico_mata_mata[time]["R32 (Classif.)"] += 1

    # FUNÇÃO INTERNA PARA JOGOS DO MATA-MATA (SEM EMPATE!)
    def jogar_mata_mata(time_A, time_B):
        res = simular_partida_mc(nome_para_id[time_A], nome_para_id[time_B], novo_modelo_copa, df_cadastro)
        if res == 2:
            return time_A
        elif res == 0:
            return time_B
        else:
            # Se der empate no tempo normal, a decisão vai para os pênaltis (50% / 50%)
            return np.random.choice([time_A, time_B])

    # EXECUÇÃO DO BRACKET ELIMINATÓRIO
    
    # 16 Avos de Final (32 times -> 16 times)
    # Para espelhar de forma justa, embaralhamos os confrontos iniciais
    np.random.shuffle(classificados_32)
    vencedores_r32 = []
    for i in range(0, 32, 2):
        vencedor = jogar_mata_mata(classificados_32[i], classificados_32[i+1])
        vencedores_r32.append(vencedor)
        historico_mata_mata[vencedor]["R16 (Oitavas)"] += 1

    # Oitavas de Final (16 times -> 8 times)
    vencedores_oitavas = []
    for i in range(0, 16, 2):
        vencedor = jogar_mata_mata(vencedores_r32[i], vencedores_r32[i+1])
        vencedores_oitavas.append(vencedor)
        historico_mata_mata[vencedor]["QF (Quartas)"] += 1

    # Quartas de Final (8 times -> 4 times)
    vencedores_quartas = []
    for i in range(0, 8, 2):
        vencedor = jogar_mata_mata(vencedores_oitavas[i], vencedores_oitavas[i+1])
        vencedores_quartas.append(vencedor)
        historico_mata_mata[vencedor]["SF (Semi)"] += 1

    # Semifinal (4 times -> 2 finalistas)
    finalistas = []
    for i in range(0, 4, 2):
        vencedor = jogar_mata_mata(vencedores_quartas[i], vencedores_quartas[i+1])
        finalistas.append(vencedor)
        historico_mata_mata[vencedor]["F (Final)"] += 1

    # Grande Final (2 times -> 1 Campeão)
    campeao = jogar_mata_mata(finalistas[0], finalistas[1])
    historico_mata_mata[campeao]["Campeão"] += 1

# TRATAMENTO DOS DADOS PARA O RELATÓRIO FINAL
dados_consolidados = []
for time, fases in historico_mata_mata.items():
    dados_consolidados.append({
        "Seleção": time,
        "Grupos (%)": (fases["R32 (Classif.)"] / N_SIMULACOES) * 100,
        "16 Avos (%)": (fases["R16 (Oitavas)"] / N_SIMULACOES) * 100,
        "Oitavas (%)": (fases["QF (Quartas)"] / N_SIMULACOES) * 100,
        "Quartas (%)": (fases["SF (Semi)"] / N_SIMULACOES) * 100,
        "Semifinal (%)": (fases["F (Final)"] / N_SIMULACOES) * 100,
        "Campeão (%)": (fases["Campeão"] / N_SIMULACOES) * 100
    })

df_dashboard = pd.DataFrame(dados_consolidados)
df_dashboard = df_dashboard.sort_values(by="Campeão (%)", ascending=False).reset_index(drop=True)

# Mascarando os números com a string de % apenas para exibição visual limpa
colunas_porcentagem = ["Grupos (%)", "16 Avos (%)", "Oitavas (%)", "Quartas (%)", "Semifinal (%)", "Campeão (%)"]
for col in colunas_porcentagem:
    df_dashboard[col] = df_dashboard[col].map(lambda x: f"{x:.1f}%")

print("PROBABILIDADES DE SUCESSO NO MATA-MATA DA COPA 2026")
print(df_dashboard.to_string(index=False))

# %%
