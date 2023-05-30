from tqdm import tqdm
from operator import add
from copy import deepcopy
from functools import reduce
from agentes.utils.problema import Problema
from agentes.resgate.planos.otimizador_genetico.individuo import Individuo
from agentes.resgate.planos.otimizador_genetico.algoritmo_genetico import AlgoritmoGenetico

class Otimizador:
    def __init__(
            self,
            n_populacao: int,
            n_geracoes: int,
            probabilidade_eventos: dict[str, float],
            tempo_disponivel: float
        ) -> None:
        self.probabilidade_eventos = probabilidade_eventos
        self.n_populacao = n_populacao
        self.n_geracoes = n_geracoes

        self.tempo_disponivel = tempo_disponivel

        self.melhor_individuo_ = None
        self.melhor_pontuacao_ = -9999999

    def evoluir(
            self,
            problema: Problema,
            caminhos_possiveis: dict[str, dict[str, list[tuple[str, int]]]]
        ) -> None:
        """Faz a evolução dos indivíduos, isto é, a otimização por meio de algoritmo genético.

        Args:
            problema (Problema): instância do problema a ser resolvido.

            caminhos_possiveis (dict[str, dict[str, list[tuple[str, int]]]]): dicionário de
                caminhos possíveis entre todas as combinações de posições de interesse.
        """
        algoritmo_genetico = AlgoritmoGenetico(
            self.probabilidade_eventos,
            caminhos_possiveis,
            problema.sinais_vitais_vitimas,
            self.tempo_disponivel
        )
        print(f"numero de vítimas para salvar: {len(problema.sinais_vitais_vitimas)}")
        populacao = algoritmo_genetico.gera_populacao_aleatoria(self.n_populacao)

        for geracao in range(1, self.n_geracoes + 1):
            print(f"Geração {geracao} de {self.n_geracoes}")

            # Faz a avaliação da população da geração atual
            self.avalia_populacao(populacao, problema.sinais_vitais_vitimas)

            # Obtém a pontuação média da população atual
            pontuacao_media = self.get_pontuacao_media(populacao)

            print(f"Media da geração {geracao}: {round(pontuacao_media, 2)}")

            # Ordena os individuos pela pontuação de modo decrescente.
            populacao = sorted(
                populacao,
                key = lambda x: x.pontuacao,
                reverse = True
            )

            print(f"{populacao[0].pontuacao} -> {populacao[0].genes['trajeto']}")
            # Salva dados do melhor individuo
            if self.melhor_pontuacao_ < populacao[0].pontuacao:
                self.melhor_individuo_ = deepcopy(populacao[0])
                # self.melhores_parametros_ = self.melhor_estimador_.get_params()
                self.melhor_pontuacao_ = self.melhor_individuo_.pontuacao

            # Evolui a geração, exceto se for a última geração.
            if geracao != self.n_geracoes:
                # Faz a evolução da populacao
                populacao = algoritmo_genetico.faz_evolucao(populacao)

        # Descreve o top 5 indivíduos no terminal
        self.__print_modelos(populacao[:5])

    def avalia_populacao(self, populacao: list[Individuo], vitimas: dict[str, list[str]]):
        """Faz a avaliação de acordo com a função de fitness dos indivíduos para cada indivíduo.

        Args:
            populacao (list[Individuo]): lista de indivíduos.
            vitimas (dict[str, list[str]]): posição e sinais vitais das vítimas.
        """
        barra_progresso = tqdm(total=len(populacao))

        for individuo in populacao:
            individuo.avalia_individuo(vitimas)
            barra_progresso.update(1)

        barra_progresso.close()

    def get_pontuacao_media(self, populacao: list[Individuo]) -> float:
        """Calcula a pontuação média de uma população.

        Args:
            populacao (list[Individuo]): indivíduos da população.

        Returns:
            float: pontuação média da população.
        """
        somatorio_pontuacao = 0

        somatorio_pontuacao = reduce(add, (individuo.pontuacao for individuo in populacao))

        return round(somatorio_pontuacao/float((len(populacao))), 2)

    def __print_modelos(self, populacao: list[Individuo]) -> None:
        """Descreve a lista dos indivíduos da população.

        Parameters:
            modelos (list): População dos modelos.
        """
        print('='*80)
        for individuo in populacao:
            print(f"{individuo.pontuacao} -> {individuo.genes['trajeto']}")
