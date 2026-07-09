CREATE TABLE estrutura (
  id_estrutura int NOT NULL,
  grupo_copa varchar(1) DEFAULT NULL,
  id_selecao int NOT NULL,
  id_confronto_grupo int NOT NULL,
  PRIMARY KEY (id_estrutura)
);
