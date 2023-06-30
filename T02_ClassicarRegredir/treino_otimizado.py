from pandas import DataFrame, read_csv
from sklearn.preprocessing import StandardScaler

from otimizador_genetico_hiperparams.modelos.arvore_decisao import (
    ArvoreDecisaoClassificacao
)

from otimizador_genetico_hiperparams.modelos.rede_neural import (
    RedeNeuralRegressao
)

from otimizador_genetico_hiperparams.modelos.modelo import Modelo
from otimizador_genetico_hiperparams.otimizador import Otimizador


class TreinamentoOtimizado:
    def __init__(self, qtd_seeds: int = 2) -> None:
        self.lista_pre_trat = []
        self.qtd_seeds = qtd_seeds

        

    def treinar_modelos(self, path_dataset: str) -> None:
        list_seeds = [(x+1)*123456789 for x in range(self.qtd_seeds)]
        metodos_avaliacao: list[str] = [
            'classificacao',
            # 'regressao'
        ]

        dataset: DataFrame = read_csv(
            "T02_ClassicarRegredir\\treino_sinais_vitais_com_label.csv",
            sep = ',',
            index_col = 'index'
        )

        x = dataset.drop(columns = [
                'pressao_sistolica',
                'pressao_diastolica',
                'gravidade',
                'classe'
            ],
            axis = 1
        )

        x = DataFrame(
            StandardScaler().fit_transform(
                    DataFrame(data = x, columns = x.columns)
                ),
            columns = x.columns,
            index = x.index
        )

        df_resumo_treino = DataFrame(
            columns = [
                'nome_algoritmo',
                'seed',
                'criterio_evolucao',
                'best_score_evolucao',
                'validacao_cruzada',
                'kfolds',
                'acuracia',
                'precisao',
                'recall',
                'f1_score',
                'confusao_final',
                'r_quadrado',
                'mae',
                'best_params',
            ]
        )

        index = 0
        for seed in list_seeds:
            for metodo_avaliacao_escolhido in metodos_avaliacao:
                if metodo_avaliacao_escolhido == 'classificacao':
                    y = dataset['classe']
                    criterios_evolucao: list[str] = [
                        'acuracia',
                        'precisao',
                        'recall',
                    ]
                    modelos_treinar: list[Modelo] = [
                        ArvoreDecisaoClassificacao,
                        # DummyClassificacao
                    ]

                    populacao = 100
                else:
                    y = dataset['gravidade']
                    criterios_evolucao: list[str] = [
                        'r_quadrado',
                        'mae',
                    ]
                    modelos_treinar: list[Modelo] = [
                        RedeNeuralRegressao,
                    ]
                    populacao = 20

                for classe_modelo_ in modelos_treinar:
                    for usa_validacao_cruzada in [True, False]:
                        for criterio in criterios_evolucao:
                            otim_gen = Otimizador(
                                X = x,
                                y = y,
                                n_geracoes = 10,
                                n_populacao = populacao,
                                classe_modelo = classe_modelo_,
                                metodo_avaliacao = metodo_avaliacao_escolhido,
                                criterio_evolucao = criterio,
                                random_state = seed,
                                por_validacao_cruzada = usa_validacao_cruzada,
                                log = True,
                                verbose = False,
                            )
                            otim_gen.evoluir()

                            df_resumo_treino.at[index, 'nome_algoritmo'] = classe_modelo_.algoritmo
                            df_resumo_treino.at[index, 'seed'] = seed
                            df_resumo_treino.at[index, 'validacao_cruzada'] = usa_validacao_cruzada
                            df_resumo_treino.at[index, 'kfolds'] = otim_gen.constantes.SPLITS_KFOLD
                            df_resumo_treino.at[index, 'criterio_evolucao'] = criterio
                            df_resumo_treino.at[index, 'best_score_evolucao'] = otim_gen.melhor_pontuacao_
                            df_resumo_treino.at[index, 'best_params'] = otim_gen.melhores_parametros_
                            metricas = otim_gen.metricas_do_melhor_

                            for metrica in metricas:
                                df_resumo_treino.at[index, metrica] = metricas[metrica]
                            index += 1

                            df_resumo_treino.to_excel(
                                "./resumo_treino_intermediario.xlsx",
                                index = False
                            )

        df_resumo_treino.to_excel(
            "./resumo_treino.xlsx",
            index = False
        )
