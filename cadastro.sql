CREATE TABLE cadastro (
  id_selecao int NOT NULL,
  nome_selecao varchar(50) NOT NULL,
  confederacao varchar(50) NOT NULL,
  ranking_fifa_pontos decimal(6,2) NOT NULL,
  valor_mercado_elenco decimal(15,2) NOT NULL,
  rating_ea_fc_geral int NOT NULL,
  rating_ea_fc_ataque int NOT NULL,
  rating_ea_fc_meio int NOT NULL,
  rating_ea_fc_defesa int NOT NULL,
  idade_media_elenco decimal(4,2) NOT NULL,
  percentual_jogadores_elite decimal(5,2) NOT NULL,
  pais_sede tinyint(1) NOT NULL,
  PRIMARY KEY (id_selecao)
  );
