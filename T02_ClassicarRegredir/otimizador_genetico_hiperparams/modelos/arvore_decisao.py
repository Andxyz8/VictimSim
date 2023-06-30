from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from .modelo import Modelo

class ArvoreDecisaoClassificacao(Modelo):
    """Representação do algoritmo DecisionTreeClassifier do sklearn.tree."""

    algoritmo = 'DecisionTree Classificacao'
    parametros_possiveis = {
        "criterion": ['gini', 'entropy', 'log_loss'],
        "max_depth": list(x for x in range(4,257,8)),
        "min_samples_split": list(x for x in range(4,64,4)),
        "min_samples_leaf": list(x for x in range(4,64,4)),
        #"max_leaf_nodes": list(x for x in range(2,11)),
        #"max_features" : list("auto", "sqrt", "log2"),
        "splitter" : ["best", "random"]
    }


    def get_modelo_construido(self):
        self.modelo = DecisionTreeClassifier(
            criterion = self.parametros['criterion'],
            max_depth = self.parametros['max_depth'],
            min_samples_split = self.parametros['min_samples_split'],
            min_samples_leaf = self.parametros['min_samples_leaf'],
            # max_leaf_nodes=self.parametros['max_leaf_nodes'],
            # max_features = self.parametros['max_features'],
            # class_weight = {0:4, 1:1},
            splitter = self.parametros['splitter'],
            random_state=self.random_state,
        )

        return self.modelo


class ArvoreDecisaoRegressao(Modelo):
    """Representação do algoritmo DecisionTreeRegressor do sklearn.tree."""

    algoritmo = 'DecisionTree Regressao'
    parametros_possiveis = {
        # "criterion": ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        "criterion": ['squared_error', 'friedman_mse', 'absolute_error'],
        "max_depth": [x for x in range(8, 129, 8)],
        "min_samples_split": [x for x in range(24, 129, 16)],
        "min_samples_leaf" : [x for x in range(24, 129, 16)],
        "max_features" : ["auto", "sqrt", "log2"],
        "splitter" : ["best", "random"],
    }


    def get_modelo_construido(self):
        self.modelo = DecisionTreeRegressor(
            criterion = self.parametros['criterion'],
            max_depth = self.parametros['max_depth'],
            min_samples_split = self.parametros['min_samples_split'],
            min_samples_leaf = self.parametros['min_samples_leaf'],
            random_state=self.random_state,
            #max_features = self.parametros['max_features'],
            #splitter = self.parametros['splitter'],
        )

        return self.modelo
