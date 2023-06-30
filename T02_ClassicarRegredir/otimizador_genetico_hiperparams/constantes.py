"""Módulo utilitário que contém a classe com os valores de constantes utilizadas
    durante a otimização genética de hiperparâmetros.
"""

class Constantes:
    """Classe utilitária que contém constantes de interesse relacionadas
        ao processo de evolução genética.

    Attributes:
        SEED (int): seed geradora das randomicidades.

        PROPORCAO_TRAIN_TEST_SPLIT (float): proporção, com relação ao todo,
            dos dados usados para treino. 

        SPLITS_KFOLD (int): número de splits para utilizar no KFold.

        N_JOBS_CROSS_VALIDATION (int): quantidade de processos para efetuar a validação cruzada.
            -1: um processo para cada núcleo do processador.
            1 ou mais: um ou mais processos para a validação cruzada.

        PORCENTAGEM_RETENCAO_POPULACAO (float): porcentagem de retenção da população
            de cada geração.

        CHANCE_MUTACAO (float): chance de ocorrer uma mutação aleatória em um indivíduo.

        CHANCE_SELECAO_ALEATORIA (float): chance de um indivíduo rejeitado ser escolhido.

        TOLERANCIA_R2_GRAFICO (float): tolerancia de desempenho de um modelo para
            construir seu gráfico de desempenho (realXpredicoes).

        LOG (bool): se True, descreve infos dos modelos em um arquivo de log.
            Não descreve, caso contrário.

        VERBOSE (bool): se True, descreve infos dos modelos na saída de texto padrão.
            Não descreve, caso contrário.
    """

    def __init__(self, seed, log, verbose) -> None:
        self.SEED: int = seed
        self.PROPORCAO_TRAIN_TEST_SPLIT: float = 0.2
        self.SPLITS_KFOLD: int = 7
        self.N_JOBS_CROSS_VALIDATION: int = 2
        self.PORCENTAGEM_RETENCAO_POPULACAO: float = 0.4
        self.CHANCE_MUTACAO: float = 0.1
        self.CHANCE_SELECAO_ALEATORIA: float = 0.05
        self.TOLERANCIA_R2_GRAFICO: float = 0.8
        self.LOG: bool = log
        self.VERBOSE: bool = verbose
