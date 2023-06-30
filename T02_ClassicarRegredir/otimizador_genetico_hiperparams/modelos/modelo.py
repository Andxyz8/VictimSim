import random
import logging
from copy import deepcopy
from abc import ABCMeta, abstractmethod
from sklearn.base import BaseEstimator
from typing import Union


class Modelo(metaclass=ABCMeta):
    """Classe abstrata que representa um modelo de machine learning do sklearn a ser evoluido.
    
    Attributes:
        parametros (dict): parâmetros a serem usados para criacao do modelo.
        pontuacao_criterio_evolucao (float): Pontuacao do modelo como criterio de evolucao.
        modelo: Modelo de machine learning do sklearn.
        metricas (dict[str, float]): métricas associadas ao desempenho do modelo.
    """
    def __init__(self, random_state: int) -> None:
        self.algoritmo: str = ''
        self.parametros: dict[str, Union[float, str]] = {}
        self.modelo: BaseEstimator = None

        self.random_state: int = random_state

        self.pontuacao_criterio_evolucao: float = 0.
        self.metricas_modelo: dict[str, float] = {}

    def set_metricas_modelo_avaliado(self, metricas_avaliadas: dict[str, float]) -> None:
        """Atribui o valor das pontuações associadas ao modelo.

        Args:
            metricas_avaliadas (dict[str, float]): valores das métricas obtidas para o modelos.
        """
        self.metricas_modelo = deepcopy(metricas_avaliadas)
        self.pontuacao_criterio_evolucao = deepcopy(
            metricas_avaliadas['pontuacao_criterio_evolucao']
        )

    def get_pontuacao_criterio_evolucao(self) -> float:
        """Retorna a pontuação do critério de evolução do modelo.

        Returns:
            (float): Pontuação do critério de evolução do modelo.
        """
        return self.pontuacao_criterio_evolucao

    def set_parametros(self, params: dict[str, Union[float, str]]) -> None:
        """Atribui os parâmetros definidos para construção do modelo.

        Parameters: 
            params (dict[str, Union[float, str]]): parametros do modelo.
        """
        self.parametros = params

    def log_print_desempenho_modelo(self, criterio_evolucao: str) -> None:
        """Descreve as informações do modelo (parametros e desempenho) no log.

        É utilizado o parâmetro ``critério_evolucao`` para saber o que será descrito no log.
        
        Parameters:
            criterio_evolucao (str): Critério de evolução aplicado.
        """
        logging.info(self.modelo.get_params())
        if criterio_evolucao  in ('r_quadrado', 'mae'):
            logging.info("R2 %.4f\tMAE  %.4f\n",
                round(self.metricas_modelo['r_quadrado']*100, 4),
                round(self.metricas_modelo['mae'], 4)
            )
        else:
            logging.info("Acuracia %.4f\tConfusao final: %.4f\n",
                round(self.metricas_modelo['acuracia']*100, 4),
                round(self.metricas_modelo['confusao_final']*100, 4)
            )

    def gera_modelo_aleatorio(self, parametros_possiveis):
        """Cria o modelo aleatoriamente, dentro dos parametros possíveis pré-estabelecidos.

        Parameters:
            parametros_possiveis (dict): parâmetros possíveis para construção do modelo aleatório.
        """
        for key in parametros_possiveis:
            self.parametros[key] = random.choice(parametros_possiveis[key])

    @abstractmethod
    def get_modelo_construido(self) -> BaseEstimator:
        """Método abstrato cria um modelo de machine learning do sklearn de acordo
            com a subclasse especificada para definir o modelo.

        BaseEstimator indica a superclasse abstrata de qualquer
            modelo de Machine Learning do sklearn.

        Returns:
            (BaseEstimator): Modelo de machine learning compilado de acordo com seus parâmetros.
        """
