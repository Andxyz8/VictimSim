from abc import ABC
import random
from .modelo import Modelo
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor


class RedeNeural(Modelo, ABC):
    def _get_estrutura_camadas(self):
        """Elabora a estrutura de camadas oculta da rede neural. Usa um valor
            gerador da quantidade de neurônios das camadas ocultas; a quantidade
            de camadas ocultas e os a quantidade (igual) para os neurônios das 
            camadas de entrada e saída.

        Returns:
            (list): estrutura das camadas da rede neural.
        """
        estrutura = [] # estrutura de camadas da rede a ser construída
        estrutura.append(self.parametros['neuronios_entrada_saida'])
        aux_camadas = []
        if random.random() >= 0.5:
            if self.parametros['qtd_camadas_ocultas'] % 2 == 0:
                for i in range(1, int(self.parametros['qtd_camadas_ocultas']/2+1)):
                    aux_camadas.append(self.parametros['qtd_neuronios_gerador']*i)

                estrutura.extend(aux_camadas)
                aux_camadas.reverse()
                estrutura.extend(aux_camadas)
            else:
                for i in range(1, int(self.parametros['qtd_camadas_ocultas']/2+1)):
                    aux_camadas.append(self.parametros['qtd_neuronios_gerador']*i)

                estrutura.extend(aux_camadas)
                estrutura.append(estrutura[-1]*2)
                aux_camadas.reverse()
                estrutura.extend(aux_camadas)
        else:
            for _ in range(self.parametros['qtd_camadas_ocultas']):
                estrutura.append(self.parametros['qtd_neuronios_gerador'])

        estrutura.append(self.parametros['neuronios_entrada_saida'])

        return estrutura


class RedeNeuralClassificacao(RedeNeural):
    """Representação do algoritmo MLPClassifier do sklearn.ensemble."""

    algoritmo = 'Rede Neural Classificacao'
    parametros_possiveis = {
        "neuronios_entrada_saida": list(x for x in range(10, 16)),
        "qtd_neuronios_gerador": list(x for x in range(20, 51)),
        # "neuronios_entrada_saida": [20,32,34,46,47,50],
        # "qtd_neuronios_gerador": [95,91,99,94,69,81],
        "qtd_camadas_ocultas": list(x for x in range (1, 11)),
        "activation": ["identity", "logistic", "tanh", "relu"],
        "solver": ["sgd", "adam"], #lbfgs
        # "learning_rate_init": list(x*0.001 for x in range(1, 11)),
        "shuffle": [True, False],
    }


    def get_modelo_construido(self):
        self.modelo = MLPClassifier(
            hidden_layer_sizes = self._get_estrutura_camadas(),
            activation = self.parametros['activation'],
            solver = self.parametros['solver'],
            shuffle = self.parametros['shuffle'],
            #alpha = self.parametros['alpha'],
            #learning_rate_init=self.parametros['learning_rate_init'],
            early_stopping=True,
            random_state=self.random_state,
            max_iter=10000,
        )

        return self.modelo


class RedeNeuralRegressao(RedeNeural):
    """Representação do algoritmo MLPRegressor do sklearn.ensemble."""

    algoritmo = 'Rede Neural Regressao'
    parametros_possiveis = {
        'neuronios_entrada_saida': list(x for x in range(2, 16)),
        'qtd_neuronios_gerador': list(x for x in range(2, 33)),
        'qtd_camadas_ocultas': list(x for x in range (1, 8)),
        'activation': ["identity", "logistic", "tanh", "relu"],
        'learning_rate': ['constant', 'invscaling', 'adaptive'],
        'solver': ["sgd", "adam"],
        'shuffle': [True, False],
    }


    def get_modelo_construido(self):
        self.modelo = MLPRegressor(
            hidden_layer_sizes = self._get_estrutura_camadas(),
            activation = self.parametros['activation'],
            solver = self.parametros['solver'],
            shuffle = self.parametros['shuffle'],
            learning_rate = self.parametros['learning_rate'],
            #alpha = self.parametros['alpha'],
            early_stopping = True,
            random_state = self.random_state,
            max_iter = 10000,
        )

        return self.modelo
    