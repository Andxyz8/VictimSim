"""
Class that holds a genetic algorithm for evolving a network.
Credit:
    A lot of those code was originally inspired by:
    http://lethain.com/genetic-algorithms-cool-name-damn-simple/
"""
import random

class Evoluidor():
    """Classe que implementa um algoritmo genético para otimização de modelos de machine learning.

    Parameters:
        parametros_possiveis (dict): parametros possiveis do modelo
        retencao (float): porcentagem da população para reter a cada nova geracao
        selecao_aleatoria (float): probabilidade de um modelo rejeitado permanecer na populacao
        chance_mutacao (float): probabilidade de um modelo sofrer uma mutacao aleatoria
    """

    def __init__(self, classe_modelo, constantes):
        self.classe_modelo = classe_modelo
        self.random_state = constantes.SEED
        self.retencao = constantes.PORCENTAGEM_RETENCAO_POPULACAO
        self.chance_mutacao = constantes.CHANCE_MUTACAO
        self.selecao_aleatoria = constantes.CHANCE_SELECAO_ALEATORIA


    def gera_populacao_aleatoria(self, qtd_individuos):
        """Cria o número de indivíduos desejados para a população com hiperparâemtros aleatórios.
        
        Parameters:
            qtd_individuos (int): Quantidade de modelos da população.
        Returns:
            (list): População dos modelos de machine learning.
        """
        populacao_modelos = []
        for _ in range(0, qtd_individuos):
            # Cria um modelo aleatório de acordo com a classe
            modelo = self.classe_modelo(random_state=self.random_state)
            modelo.gera_modelo_aleatorio(self.classe_modelo.parametros_possiveis)

            populacao_modelos.append(modelo)

        return populacao_modelos


    def cruza_modelos(self, modelo_mae, modelo_pai):
        """Cria dois modelos modelos_filhos com características dos seus pais.

        Parameters:
            modelo_mae (dict): Parâmetros do modelo mãe.
            modelo_pai (dict): Parâmetros do modelo pai.
        Returns:
            (list): Dois modelos com características dos pais.
        """
        modelos_modelos_filhos = []
        # Loop que representa a quantidade de modelos_filhos gerados.
        for _ in range(2):

            filho = {}

            # Loop pelos parâmetros dos pais para escolher aleatoriamente características para o filho.
            for param in self.classe_modelo.parametros_possiveis:
                filho[param] = random.choice([modelo_mae.parametros[param], modelo_pai.parametros[param]])

            # Cria o modelo do algoritmo de ML.
            modelo = self.classe_modelo(random_state=self.random_state)
            modelo.set_parametros(filho)

            # Chance de mutação dos parâmetros de modo aleatório.
            if self.chance_mutacao > random.random():
                modelo = self.aplica_mutacao(modelo)

            modelos_modelos_filhos.append(modelo)

        return modelos_modelos_filhos

    def aplica_mutacao(self, modelo):
        """Aleatoriamente faz a mutação de alguma parte do modelo.

        Parameters:
            modelo (dict): Modelo para efetuar a mutação.
        Returns:
            (Modelo): Modelo com a mutação aleatória aplicada.
        """
        # Escolhe um parâmetro aleatório dentro dos parâmetros possíveis.
        parametro_mutacao = random.choice(list(self.classe_modelo.parametros_possiveis.keys()))

        # Faz a mutacao em cima do parâmetro escolhido
        modelo.parametros[parametro_mutacao] = random.choice(self.classe_modelo.parametros_possiveis[parametro_mutacao])

        return modelo

    def faz_evolucao(self, populacao):
        """Evolui uma populacao de modelos de machine learning.
        
        Parameters:
            populacao (list): Lista com os individuos da populacao de modelos
        Returns:
            (list): Geração seguinte da populacao de modelos
        """
        # Obtém a pontuacao para cada um dos modelos da população
        pontuacao_modelos = [(modelo.get_pontuacao_criterio_evolucao(), modelo) for modelo in populacao]

        # Ordena as pontuações de forma crescente
        pontuacao_modelos = [x[1] for x in sorted(pontuacao_modelos, key=lambda x: x[0], reverse=True)]

        # Seleciona a quantidade que deve ser mantida para a próxima geração
        qtd_modelos_mantidos = int(len(pontuacao_modelos)*self.retencao)

        # Os modelos geradores serão todos aqueles que foram mantidos
        modelos_geradores = pontuacao_modelos[:qtd_modelos_mantidos]

        # Aleatoriamente mantém alguns daqueles que foram descartados
        for individuo in pontuacao_modelos[qtd_modelos_mantidos:]:
            if (self.selecao_aleatoria > random.random()):
                modelos_geradores.append(individuo)

        # Quantidade de crianças que devem ser geradas
        qtd_modelos_geradores = len(modelos_geradores)
        qtd_modelos_desejada = len(populacao) - qtd_modelos_geradores
        modelos_filhos = []

        # Cria os filhos com as características dos modelos remanescentes.
        while (len(modelos_filhos) < qtd_modelos_desejada):

            # Seleciona dois modelos geradores aleatórios.
            modelo_pai = random.randint(0, qtd_modelos_geradores-1)
            modelo_mae = random.randint(0, qtd_modelos_geradores-1)

            # Verifica se os modelos são diferentes
            if (modelo_pai != modelo_mae):
                modelo_pai = modelos_geradores[modelo_pai]
                modelo_mae = modelos_geradores[modelo_mae]

                # Cruza os modelos selecionados e retorna os filhos.
                filhos = self.cruza_modelos(modelo_pai, modelo_mae)

                # Acrescenta um filho por vez até a quantidade desejada.
                for filho in filhos:
                    if (len(modelos_filhos) < qtd_modelos_desejada):
                        modelos_filhos.append(filho)

        modelos_geradores.extend(modelos_filhos)

        return modelos_geradores