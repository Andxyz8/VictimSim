from .avaliador import Avaliador
from .constantes import Constantes
from .evoluidor import Evoluidor
from .modelos.modelo import Modelo
import logging
from copy import deepcopy
from tqdm import tqdm


class Otimizador:
    """Classe que controla a otimização dos hiperparâmetros de forma genética.

    Attributes:
        X (DataFrame): dados previsores/independentes.

        y (DataFrame): dados classe/dependentes.

        n_geracoes (int): quantidade de gerações.

        n_populacao (int): número de individuos na população.

        classe_modelo (Modelo): classe do modelo a ser evoluído, necessário para ser genérico.

        metodo_avaliacao (str): método de avaliação dos dados passados. Pode ser:

                'regressao': valores contínuos para y.

                'classificacao': valores categóricos (classes) para y.

                'regressao_dps_classificacao': após a regressão resultados convertidos
                        para valores categóricos (classes).

        criterio_evolucao (str): critério utilizado como avaliação para seleção dos indivíduos
            com os melhores hiperparâmetros. Default 'r_quadrado'.

        random_state (int): seed geradora das aleatóriedades do pipeline. Default 42.

        por_validacao_cruzada (bool): se True a avaliação é feita por validação cruzada,
            caso contrário por proporção. Default False.

        log (bool): se True gera o log da otimização, não gera, caso contrário. Default True.

        verbose (bool): se True descreve algumas informações na saída de texto padrão,
            caso contrário não descreve. Default True.
    """
    def __init__(
            self,
            X,
            y,
            n_geracoes: int,
            n_populacao: int,
            classe_modelo: Modelo,
            metodo_avaliacao: str = 'acuracia',
            criterio_evolucao: str = 'r_quadrado',
            random_state = 42,
            por_validacao_cruzada = False,
            log = True,
            verbose = True
        ) -> None:
        self.n_geracoes = n_geracoes
        self.n_populacao = n_populacao
        self.classe_modelo = classe_modelo
        self.metodo_avaliacao = metodo_avaliacao
        self.criterio_evolucao = criterio_evolucao
        self.por_validacao_cruzada = por_validacao_cruzada
        self.constantes = Constantes(
            seed = random_state,
            log = log,
            verbose = verbose
        )
        self.evoluidor = Evoluidor(self.classe_modelo, self.constantes)
        self.avaliador = Avaliador(
            x_dados = X,
            y_dados = y,
            metodo_avaliacao = self.metodo_avaliacao,
            criterio_evolucao = self.criterio_evolucao,
            por_validacao_cruzada = por_validacao_cruzada,
            constantes = self.constantes
        )
        self.melhor_estimador_ = None
        self.melhor_pontuacao_ = 0.
        self.melhores_parametros_ = {}
        self.metricas_do_melhor_ = {}

        if self.constantes.LOG:
            # Configuração do logging.
            logging.basicConfig(
                format='%(asctime)s - %(levelname)s - %(message)s',
                datefmt='%m/%d/%Y %I:%M:%S %p',
                level=logging.DEBUG,
                filename='log.txt'
            )


    def evoluir(self):
        """ Pipeline de evolução dos hiperparâmetros modelo pela quantidade de gerações definida.
        """
        todos_modelos: list[Modelo] = self.evoluidor.gera_populacao_aleatoria(self.n_populacao)

        # Evolui a geração
        for geracao in range(self.n_geracoes):
            #Escreve no log informações gerais do andamento da otimização
            if self.constantes.LOG:
                logging.info("***Geracao (%s) %d de %d***",
                    self.classe_modelo.algoritmo,
                    geracao + 1,
                    self.n_geracoes
                )
                logging.info("CONDICOES: criterio = %s, validacao_cruzada = %s",
                    self.criterio_evolucao,
                    str(self.por_validacao_cruzada)
                )

            # Faz o treinamento e obtém o desempenho de todos os modelos da população atual
            self.avaliar_modelos(todos_modelos)

            # Obtém a pontuação média da população atual
            pontuacao_media = self.avaliador.get_pontuacao_media(todos_modelos)

            # Escreve no log a média da pontuação da geração atual
            if self.constantes.LOG:
                logging.info("Media Geracao %s: %.2f",
                    self.criterio_evolucao,
                    pontuacao_media
                )

            # Ordena os modelos pela pontuação de modo decrescente.
            todos_modelos = sorted(
                todos_modelos,
                key = lambda x: x.pontuacao_criterio_evolucao,
                reverse = True
            )

            # Se for a geração 0, salva os dados do melhor estimador de qualquer maneira
            if not geracao:
                self.melhor_estimador_ = deepcopy(todos_modelos[0].modelo)
                self.melhores_parametros_ = self.melhor_estimador_.get_params()
                self.melhor_pontuacao_ = deepcopy(todos_modelos[0].pontuacao_criterio_evolucao)
                self.metricas_do_melhor_ = deepcopy(todos_modelos[0].metricas_modelo)
            # Se geração != 0, salva os dados do melhor estimador, se for melhor que aquele salvo
            else:
                if self.melhor_pontuacao_ < todos_modelos[0].pontuacao_criterio_evolucao:
                    self.melhor_estimador_ = deepcopy(todos_modelos[0].modelo)
                    self.melhores_parametros_ = self.melhor_estimador_.get_params()
                    self.melhor_pontuacao_ = deepcopy(todos_modelos[0].pontuacao_criterio_evolucao)
                    self.metricas_do_melhor_ = deepcopy(todos_modelos[0].metricas_modelo)


            # Escreve no log a o desempenho de cada modelo da geração
            if self.constantes.LOG:
                self.__print_modelos(todos_modelos)

            # Evolui a geração, exceto se for a última geração.
            if geracao != self.n_geracoes - 1:
                # Faz a evolução do modelo
                todos_modelos = self.evoluidor.faz_evolucao(todos_modelos)

    def avaliar_modelos(self, modelos: list[Modelo]):
        """Faz a avaliacao dos modelos da populacao atual.

        Parameters:
            modelos (list): Populacao de modelos atual.
        """
        barra_progresso = tqdm(total = len(modelos))

        for modelo in modelos:
            modelo.set_metricas_modelo_avaliado(
                self.avaliador.avaliar_modelo(modelo)
            )
            barra_progresso.update(1)

        barra_progresso.close()

    def __print_modelos(self, modelos: list[Modelo]):
        """Descreve a lista dos modelos da população no log.

        Parameters:
            modelos (list): População dos modelos.
        """
        logging.info('-'*80)
        for modelo in modelos:
            modelo.log_print_desempenho_modelo(self.criterio_evolucao)
