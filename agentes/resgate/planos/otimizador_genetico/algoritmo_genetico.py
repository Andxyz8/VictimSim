from random import choice, random, randint
from tqdm import tqdm
from agentes.resgate.planos.otimizador_genetico.individuo import Individuo


class AlgoritmoGenetico:
    def __init__(
            self,
            probabilidade_eventos: dict[str, float],
            caminhos: dict[str, dict[str, list[tuple[str, int]]]],
            vitimas: dict[str, list[str]],
            tempo_disponivel: float
        ) -> None:
        self.probabilidade_eventos = probabilidade_eventos
        self.caminhos = caminhos
        self.vitimas = vitimas
        self.tempo_disponivel = tempo_disponivel
        self.max_subcaminhos = len(self.vitimas.keys()) + 1

    def aplica_mutacao(self, individuo: Individuo) -> Individuo:
        """Aplica a mutação em uma das características mutáveis do indivíduo.

        Args:
            individuo (Individuo): indivíduo a ser mutado.

        Returns:
            Individuo: Indivíduo com a mutação aplicada.
        """
        # Escolhe um parâmetro aleatório dentro dos parâmetros possíveis.
        parametro_mutacao = choice(list(individuo.parametros_possiveis.keys()))

        # Faz a mutacao em cima do parâmetro escolhido
        individuo.genes[parametro_mutacao] = choice(
            individuo.parametros_possiveis[parametro_mutacao]
        )
        if 'gravidade' in parametro_mutacao:
            individuo.genes['peso_custo'] = 1 - individuo.genes[parametro_mutacao]
        else:
            individuo.genes['peso_gravidade'] = 1 - individuo.genes[parametro_mutacao]

        return individuo

    def crossover(self, pai: Individuo, mae: Individuo) -> list[Individuo]:
        """Efetua o crossover das características de dois Indivíduos para geração de outro.

        Args:
            pai (Individuo): objeto Individuo considerado pai.
            mae (Individuo): objeto Individuo considerado mae.

        Returns:
            list[Individuo]: uma lista contendo dois indivíduos criados pelo pai e mae passados.
        """
        filhos = []
        # Loop que representa a quantidade de filhos a serem gerados.
        for _ in range(2):
            caracteristicas_filho = {}

            # Escolhe aleatoriamente características para o filho.
            for param in pai.parametros_possiveis:
                caracteristicas_filho[param] = choice([mae.genes[param], pai.genes[param]])

            # Gera um filho com uma nova sequência de sub-caminhos aleatório
            filho = Individuo(self.caminhos, self.tempo_disponivel, self.max_subcaminhos)
            filho.gera_individuo_aleatorio()

            # Atribui as características de peso dos pais para a função de fitness do filho
            filho.set_caracteristicas(caracteristicas_filho)

            # Avalia a pontuação deste indivíduo
            filho.avalia_individuo(self.vitimas)

            # Se a pontuação do filho for menor que a dos pais, descarta o trajeto do filho
            if filho.pontuacao < pai.pontuacao and filho.pontuacao < mae.pontuacao:
                trajeto = mae.genes['trajeto']
                if pai.pontuacao > mae.pontuacao:
                    trajeto = pai.genes['trajeto']

                # Atribui o trajeto com maior pontuação entre os pais
                filho.set_trajeto(trajeto)

            # Chance de mutação dos parâmetros de modo aleatório.
            if self.probabilidade_eventos['chance_mutacao'] > random():
                filho = self.aplica_mutacao(filho)

            # Insere o filho gerado na lista de filhos destes pais
            filhos.append(filho)

        return filhos

    def faz_evolucao(self, populacao: list[Individuo]) -> list[Individuo]:
        """Realiza a evolução dos indivíduos de uma população.

        Args:
            populacao (list[Individuo]): lista de indivíduos da população.

        Returns:
             list[Individuo]: lista de indivíduos (população) evoluída.
        """
        # Obtém a pontuacao para cada um dos individuos da população
        pontuacao_individuos = [
            {'pontuacao': individuo.pontuacao, 'individuo': individuo}
            for individuo in populacao
        ]

        # Ordena as pontuações de forma crescente
        individuos_ordenados = [
            x['individuo'] for x in sorted(
                pontuacao_individuos,
                key = lambda x: x['pontuacao'],
                reverse = True
            )
        ]

        # Seleciona a quantidade que deve ser mantida para a próxima geração
        qtd_individuos_mantidos = int(
            self.probabilidade_eventos['percentagem_retencao_populacao']
            * int(len(individuos_ordenados))
        )

        # Os individuos geradores serão todos aqueles que foram mantidos
        individuos_geradores = individuos_ordenados[:qtd_individuos_mantidos]

        # Aleatoriamente mantém alguns daqueles que foram descartados
        for individuo in individuos_ordenados[qtd_individuos_mantidos:]:
            if self.probabilidade_eventos['chace_selecao_aleatoria'] > random():
                individuos_geradores.append(individuo)

        # Quantidade de crianças que devem ser geradas
        qtd_individuos_geradores = len(individuos_geradores)
        qtd_individuos_desejada = len(populacao) - qtd_individuos_geradores
        individuos_filhos = []

        print("GERANDO FILHOS PARA GERAÇÃO SEGUINTE")
        barra_progresso = tqdm(total = qtd_individuos_desejada)
        # Cria os filhos com as características dos individuos remanescentes.
        while len(individuos_filhos) < qtd_individuos_desejada:
            # Seleciona dois individuos geradores aleatórios.
            pos_individuo_pai = randint(0, qtd_individuos_geradores-1)
            pos_individuo_mae = randint(0, qtd_individuos_geradores-1)

            # Verifica se os individuos são diferentes
            if pos_individuo_pai != pos_individuo_mae:
                individuo_pai = individuos_geradores[pos_individuo_pai]
                individuo_mae = individuos_geradores[pos_individuo_mae]

                # Cruza os individuos selecionados e retorna os filhos.
                filhos = self.crossover(individuo_pai, individuo_mae)

                # Acrescenta um filho por vez até a quantidade desejada.
                for filho in filhos:
                    if len(individuos_filhos) < qtd_individuos_desejada:
                        individuos_filhos.append(filho)
                        barra_progresso.update(1)
        barra_progresso.close()
        # Inclui os filhos gerados na nova população
        individuos_geradores.extend(individuos_filhos)

        # Retorna a população evoluída
        return individuos_geradores

    def gera_populacao_aleatoria(self, qtd_individuos: int) -> list[Individuo]:
        """Gera uma população de indivíduos com trajetos aleatorios.

        Args:
            qtd_individuos (int): quantidade de indivíduos para gerar a população.

        Returns:
            list[Individuo]: lista de indivíduos aleatórios gerados.
        """
        populacao_gerada = []
        print("GERANDO INDIVIDUOS ALEATÓRIOS")
        barra_progresso = tqdm(total = qtd_individuos)
        for _ in range(qtd_individuos):
            individuo = Individuo(self.caminhos, self.tempo_disponivel, self.max_subcaminhos)
            individuo.gera_individuo_aleatorio()
            populacao_gerada.append(individuo)
            individuo = None
            barra_progresso.update(1)

        barra_progresso.close()

        return populacao_gerada
