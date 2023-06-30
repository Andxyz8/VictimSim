"""Módulo de avaliação dos modelos para o otimizador genético de hiperparâmetros.
"""
from .modelos.modelo import Modelo
from .constantes import Constantes
from operator import add
from functools import reduce
from matplotlib import pyplot as plt
from numpy import (
    polyfit,
    poly1d,
    where,
    ndarray
)
from sklearn.metrics import (
    r2_score,
    accuracy_score,
    mean_absolute_error,
    confusion_matrix,
    recall_score,
    precision_score
)
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate, train_test_split

class Avaliador():
    """Classe que avalia modelos de machine learning de acordo com o conjunto de dados passado.
    
    Parameters:
        metodo_avaliacao: formas de avaliar os modelos. Maneiras disponíveis:\n
                'classificacao': usa métricas de modelos de classificação.\n
                'regressao': usa métricas de modelos de regressão.\n
                'regressao_dps_classificacao': usa métricas de regressão e classificação.

        pontuacao_criterio_evolucao: de acordo com o atributo self.criterio_evolucao.\n
                'acuracia': acurácia do modelo avaliado.\n
                'confusao_final': diferença da confusão entre as classes.\n
                'r_quadrado': r2 do modelo avaliado.\n
                'mae': erro absoluto médio do modelo avaliado.
    """
    def __init__(
        self,
        x_dados,
        y_dados,
        metodo_avaliacao: str,
        criterio_evolucao: str,
        por_validacao_cruzada: bool,
        constantes: Constantes
    ) -> None:
        self.constantes: Constantes = constantes
        self.metodo_avaliacao = metodo_avaliacao
        self.criterio_evolucao = criterio_evolucao
        self.por_validacao_cruzada = por_validacao_cruzada

        self.x_dados = x_dados
        self.y_dados = y_dados

        if not self.por_validacao_cruzada:
            self.x_treino, self.x_teste, self.y_treino, self.y_teste = train_test_split(
                self.x_dados,
                self.y_dados,
                shuffle = True,
                test_size = self.constantes.PROPORCAO_TRAIN_TEST_SPLIT,
                random_state = self.constantes.SEED,
                # stratify = self.y
            )

    def avaliar_modelo(self, modelo: Modelo) -> dict[str, float]:
        """Treina o modelo e retorna os dados das métricas do seu desempenho
            de acordo com o metodo de avaliação escolhido.

        Parameters:
            modelo (Modelo): modelo a ser treinado e avaliado.

        Returns:
            dict[str, float]: dicionário contendo todas as métricas avaliadas.
        """
        modelo: BaseEstimator = modelo.get_modelo_construido()
        metricas: dict = {}
        if not self.por_validacao_cruzada:
            try:
                modelo = modelo.fit(self.x_treino, self.y_treino)
            except ValueError:
                return {
                    'pontuacao_criterio_evolucao': 0.,
                    'r_quadrado': 0.,
                    'mae': 0.,
                }

            predicoes = modelo.predict(self.x_teste)

            if self.metodo_avaliacao == 'regressao_dps_classificacao':
                metricas = self.__obtem_metricas_classificao_dps_da_regressao(predicoes)

            if self.metodo_avaliacao == 'classificacao':
                metricas = self.__obtem_metricas_classificacao(predicoes)

            if self.metodo_avaliacao == 'regressao':
                metricas = self.__obtem_metricas_regressao(predicoes)
        else:
            if self.metodo_avaliacao == 'regressao_dps_classificacao':
                metricas = self.__obtem_metricas_cross_classificao_dps_da_regressao(modelo)

            if self.metodo_avaliacao == 'classificacao':
                metricas = self.__obtem_metricas_cross_classificacao(modelo)

            if self.metodo_avaliacao == 'regressao':
                metricas = self.__obtem_metricas_cross_regressao(modelo)

        metricas['pontuacao_criterio_evolucao'] = self.__obtem_pontuacao_criterio_evolucao(
            metricas
        )

        return metricas

    def __gera_grafico_real_vs_predicao(self, modelo, previsto, real, r2, mae):
        """Gera o gráfico dos valores rais pelas predições do modelo avaliado.

            Inclui as retas de real x real (vermelho) e reta de tendência (verde).
        """
        plt.plot(real, real, color='red')
        plt.scatter(real, previsto)

        z = polyfit(real, previsto, 1)
        polinomio_tendencia = poly1d(z)
        plt.plot(real, polinomio_tendencia(real),"green")


        plt.title(f"R2 {round(r2,4)} MAE {round(mae, 4)}")
        plt.xlabel("Real (lotes/mes)")
        plt.ylabel("Predicao (lotes/mes)")

        plt.savefig(f"imgs\\{str(type(modelo)).split('.', maxsplit = 1)[-1].replace('>', '')} R2 {round(r2,4)} MAE {round(mae, 4)}.png")
        plt.clf()

    def get_pontuacao_media(self, populacao_modelos: list[Modelo]) -> float:
        """Obtém o desempenho médio da população de modelos de acordo
            com o critério de evolução utilzado.

        Parameters:
            populacao_modelos (list[Modelo]): População de modelos.
        Returns:
            float: Desempenho médio da população de modelos.
        """
        somatorio_pontuacao = 0

        somatorio_pontuacao = reduce(
            add,
            (modelo.get_pontuacao_criterio_evolucao() for modelo in populacao_modelos))

        if self.criterio_evolucao in ('mae', 'confusao_final'):
            somatorio_pontuacao = somatorio_pontuacao*(-1)
        else:
            somatorio_pontuacao = somatorio_pontuacao*100

        return somatorio_pontuacao/float(len(populacao_modelos))

    def __obtem_metricas_regressao(self, predicoes: ndarray) -> dict[str, float]:
        metricas: dict[str, float] = {
            'pontuacao_criterio_evolucao': 0.,
            'r_quadrado': 0.,
            'mae': 0.,
        }

        metricas['r_quadrado'] = self.__obtem_r_quadrado(self.y_teste, predicoes)

        metricas['mae'] = self.__obtem_mae(self.y_teste, predicoes)

        # if resultado_r2 >= self.constantes.TOLERANCIA_R2_GRAFICO:
        #     self.__gera_grafico_real_vs_predicao(
        #         modelo,
        #         y_predicao,
        #         self.y_teste,
        #         resultado_r2*100,
        #         resultado_mae
        #     )
        return metricas

    def __obtem_metricas_cross_regressao(self, modelo) -> dict[str, float]:
        metricas: dict[str, float] = {
            'pontuacao_criterio_evolucao': 0.,
            'r_quadrado': 0.,
            'mae': 0.,
        }
        try:
            resultado = cross_validate(
                modelo,
                self.x_dados,
                self.y_dados,
                scoring = ['r2', 'neg_mean_absolute_error'],
                n_jobs = self.constantes.N_JOBS_CROSS_VALIDATION,
                cv = KFold(
                    n_splits = self.constantes.SPLITS_KFOLD,
                    shuffle = True,
                    random_state = self.constantes.SEED,
                ),
                # error_score='raise'
            )
        except ValueError:
            return {
                'pontuacao_criterio_evolucao': 0.,
                'r_quadrado': -99.,
                'mae': 99.,
        }

        metricas['r_quadrado'] = resultado['test_r2'].mean()
        metricas['mae'] = resultado['test_neg_mean_absolute_error'].mean()*(-1)

        return metricas

    def __obtem_metricas_classificacao(self, predicoes: ndarray) -> dict[str, float]:
        metricas: dict[str, float] = {
            'pontuacao_criterio_evolucao': 0.,
            'acuracia': 0.,
            'confusao_final': 0.,
            'precisao': 0.,
            'recall': 0.,
            'f1_score': 0.,
        }

        metricas['acuracia'] = self.__obtem_acuracia(
            self.y_teste,
            predicoes
        )
        metricas['recall'] = self.__obtem_recall(
            self.y_teste,
            predicoes
        )
        metricas['precisao'] = self.__obtem_precisao(
            self.y_teste,
            predicoes
        )
        metricas['f1_score'] = self.__obtem_f1_score(
            metricas['recall'],
            metricas['precisao']
        )
        metricas['confusao_final'] = self.__obtem_confusao_final(
            self.y_teste,
            predicoes
        )

        return metricas

    def __obtem_metricas_cross_classificacao(self, modelo) -> dict[str, float]:
        metricas: dict[str, float] = {
            'pontuacao_criterio_evolucao': 0.,
            'acuracia': 0.,
            'confusao_final': 0.,
            'precisao': 0.,
            'recall': 0.,
            'f1_score': 0.,
        }
        resultado = cross_validate(
            modelo,
            self.x_dados,
            self.y_dados,
            scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted'],
            n_jobs = self.constantes.N_JOBS_CROSS_VALIDATION,
            cv = KFold(
                n_splits = self.constantes.SPLITS_KFOLD,
                shuffle = True,
                random_state = self.constantes.SEED,
            ),
        )

        metricas['acuracia'] = resultado['test_accuracy'].mean()
        metricas['precisao'] = resultado['test_precision_weighted'].mean()
        metricas['recall'] = resultado['test_recall_weighted'].mean()
        metricas['f1_score'] = resultado['test_f1_weighted'].mean()

        return metricas

    def __obtem_metricas_classificao_dps_da_regressao(
        self,
        predicoes: ndarray
    ) -> dict[str, float]:
        metricas: dict[str, float] = {
            'pontuacao_criterio_evolucao': 0., # Qualquer uma daquelas pre-definidas.
            'r_quadrado': 0., # Coeficiente de ajuste à regressão.
            'mae': 0., # Erro medio absoluto.
            'acuracia': 0., # Taxa total de acertos das predições.
            'confusao_final': 0., # Diferença absoluta entre o recall positivo e negativo.
            'precisao': 0., # Taxa de acertos das predições positivas.
            'recall': 0., # Taxa de acertos da realidade positiva.
            'f1_score': 0., # Média harmônica entre recall e precisão.
        }

        metricas['r_quadrado'] = self.__obtem_r_quadrado(
            self.y_teste,
            predicoes
        )

        metricas['mae'] = self.__obtem_mae(
            self.y_teste,
            predicoes
        )

        y_predicao_classificacao_binaria = where(predicoes >= 0, 1, 0)
        y_verdade_classificacao_binaria = where(self.y_teste >= 0, 1, 0)

        metricas['acuracia'] = self.__obtem_acuracia(
            y_verdade_classificacao_binaria,
            y_predicao_classificacao_binaria
        )
        metricas['recall'] = self.__obtem_recall(
            y_verdade_classificacao_binaria,
            y_predicao_classificacao_binaria
        )
        metricas['precisao'] = self.__obtem_precisao(
            y_verdade_classificacao_binaria,
            y_predicao_classificacao_binaria
        )
        metricas['f1_score'] = self.__obtem_f1_score(
            metricas['recall'],
            metricas['precisao']
        )
        metricas['confusao_final'] = self.__obtem_confusao_final(
            y_verdade_classificacao_binaria,
            y_predicao_classificacao_binaria
        )

        return metricas

    def __obtem_metricas_cross_classificao_dps_da_regressao(self, modelo) -> dict[str, float]:
        metricas: dict[str, float] = {
            'pontuacao_criterio_evolucao': 0., # Qualquer uma daquelas pre-definidas.
            'r_quadrado': 0., # Coeficiente de ajuste à regressão.
            'mae': 0., # Erro medio absoluto.
            'acuracia': 0., # Taxa total de acertos das predições.
            'confusao_final': 0., # Diferença absoluta entre o recall positivo e negativo.
            'precisao': 0., # Taxa de acertos das predições positivas.
            'recall': 0., # Taxa de acertos da realidade positiva.
            'f1_score': 0., # Média harmônica entre recall e precisão.
        }
        resultado = cross_validate(
            modelo,
            self.x_dados,
            self.y_dados,
            scoring=['r2', 'neg_mean_absolute_error'],
            n_jobs=self.constantes.N_JOBS_CROSS_VALIDATION,
            cv = KFold(
                n_splits=self.constantes.SPLITS_KFOLD,
                shuffle=True,
                random_state=self.constantes.SEED,
            ),
            # error_score='raise'
        )
        metricas['r_quadrado'] = resultado['test_r2'].mean()
        metricas['mae'] = resultado['test_neg_mean_absolute_error'].mean()*(-1)

        return metricas

    def __obtem_pontuacao_criterio_evolucao(self, metricas_avaliadas: dict[str, float]) -> float:
        """Retorna o valor da métrica de modo apropriado para realizar o processo de evolução.

        Args:
            metricas (dict[str, float]): metricas do modelo avaliado.

        Returns:
            float: valor da métrica utilizada para realizar o processo de evolução.
        """
        # QUANTO MENOR MELHOR, PORTANTO *(-1)
        if self.criterio_evolucao in ('mae', 'confusao_final'):
            return metricas_avaliadas[self.criterio_evolucao] * (-1)
        return metricas_avaliadas[self.criterio_evolucao]

    def __obtem_acuracia(self, y_verdade: ndarray, y_predicao: ndarray) -> float:
        """Retorna a acurácia do modelo avaliado.

        - Taxa de acertos sobre o total de predições realizadas.

        Args:
            y_verdade (ndarray): classificações corretas.
            y_predicao (ndarray): predições do modelo.

        Returns:
            float: valor da acurácia do modelo.
        """
        return accuracy_score(y_verdade, y_predicao)

    def __obtem_r_quadrado(self, y_verdade: ndarray, y_predicao: ndarray) -> float:
        """Retorna o r_quadrado do modelo avaliado.

        - Coeficiente de determinação.

        Args:
            y_verdade (ndarray): valores corretos.
            y_predicao (ndarray): previsões do modelo.

        Returns:
            float: valor do r_quadrado do modelo.
        """
        return r2_score(y_verdade, y_predicao)

    def __obtem_mae(self, y_verdade: ndarray, y_predicao: ndarray) -> float:
        """Retorna o erro médio absoluto (mae) do modelo avaliado.

        Args:
            y_verdade (ndarray): valores corretos.
            y_predicao (ndarray): previsões do modelo.

        Returns:
            float: valor do mae do modelo.
        """
        return mean_absolute_error(y_verdade, y_predicao)

    def __obtem_precisao(self, y_verdade: ndarray, y_predicao: ndarray) -> float:
        """Retorna a precisão do modelo avaliado.

        - Taxa de verdadeiros positivos previstos que foram previstos corretamente.

        Args:
            y_verdade (ndarray): classificações corretas.
            y_predicao (ndarray): predições do modelo.
            
        Returns:
            float: valor da precisão do modelo.
        """
        return precision_score(y_verdade, y_predicao, zero_division = 1, average = 'micro')

    def __obtem_recall(self, y_verdade: ndarray, y_predicao: ndarray) -> float:
        """Retorna o recall (revocação) do modelo avaliado.

        - Taxa de verdadeiros positivos total que foram previstos corretamente.

        Args:
            y_verdade (ndarray): classificações corretas.
            y_predicao (ndarray): predições do modelo.
            
        Returns:
            float: valor do recall (revocação) do modelo.
        """
        return recall_score(y_verdade, y_predicao, zero_division = 1, average = 'micro')

    def __obtem_f1_score(self, recall: float, precisao: float) -> float:
        """Retorna o f1_score do modelo avaliado.
        
        - Média harmônica entre o valor do recall e precisao.

        Args:
            recall (float): valor do recall do modelo avaliado.
            precisao (float): valor da precisao do modelo avaliado

        Returns:
            float: f1_score do modelo avaliado.
        """
        return (2*precisao*recall)/(precisao+recall)

    def __obtem_confusao_final(self, y_verdade, y_predicao) -> float:
        """Retorna o valor da confusão entre as classes do modelo avaliado.
        
        - Diferença entre a taxa de predições incorretas da classificação binária.

        Args:
            y_verdade (ndarray): classificações corretas.
            y_predicao (ndarray): predições do modelo.

        Returns:
            float: valor da confusão entre as classes.
        """
        matriz_confusao = self.__obtem_matriz_confusao(
            y_verdade,
            y_predicao
        )
        confusao_neg = matriz_confusao[0, 1]/(matriz_confusao[0, 0] + matriz_confusao[0, 1])
        confusao_pos = matriz_confusao[1, 0]/(matriz_confusao[1, 1] + matriz_confusao[1, 0])

        return abs(confusao_pos - confusao_neg)

    def __obtem_matriz_confusao(self, y_verdade, y_predicao) -> list[list[int]]:
        """Retorna a matriz de confusão das predições do modelo avaliado.

        Args:
            y_verdade (ndarray): classificações corretas.
            y_predicao (ndarray): predições do modelo.

        Returns:
            list[list[int]]: matriz de confusão das predições do modelo avaliado.
        """
        # VN = [0, 0]; FP = [0, 1];
        # FN = [1, 0]; VP = [1, 1];
        return confusion_matrix(y_verdade, y_predicao)
