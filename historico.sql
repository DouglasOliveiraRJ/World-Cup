CREATE TABLE historico (
  id_partida int NOT NULL AUTO_INCREMENT,
  data_partida date DEFAULT NULL,
  id_selecao_mandante bigint DEFAULT NULL,
  id_selecao_visitante bigint DEFAULT NULL,
  gols_mandante bigint DEFAULT NULL,
  gols_visitante bigint DEFAULT NULL,
  tipo_competicao text,
  fator_campo text,
  resultado text,
  PRIMARY KEY (id_partida)
);
