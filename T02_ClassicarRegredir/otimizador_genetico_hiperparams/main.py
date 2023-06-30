from modelos.adaboost import *
from modelos.arvore_decisao import *
from modelos.floresta_aleatoria import *
from modelos.gradient_boosting import *
from modelos.knn import *
from modelos.naive_bayes import *
from modelos.rede_neural import *
from modelos.regressao_logistica import *
from modelos.svm import *
from modelos.sgd import *
from otimizador import Otimizador
from dados import Dados


def executar_otimizacao_bru(n_geracoes, n_populacao, SEED=42, path_dataset_analisecredito = 'datasets\\sorted_data_sub.xlsx'):
    """Executa as evoluções definidas neste escopo para a Bru, por determinadas gerações e população."""

    # DATASET DA ANÁLISE DE CRÉDITO
    # path_dataset_analisecredito = 'datasets\\sorted_data_sub.xlsx'

    dados_analise_credito = Dados(
        caminho_arquivo_dados=path_dataset_analisecredito,
        nome_dataset='analise_credito',
        padronizar=True,
    )

    dados_previsores, dados_alvo = dados_analise_credito.get_conjuntos_completos_separados()

    from modelos.arvore_decisao import ArvoreDecisaoClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=ArvoreDecisaoClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False,
        log=True,
        verbose=True,
    )
    otimizador.evoluir()

    '''from modelos.rede_neural import RedeNeuralClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=RedeNeuralClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.floresta_aleatoria import FlorestaAleatoriaClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=FlorestaAleatoriaClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.gradient_boosting import GradientBoostingClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=GradientBoostingClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.adaboost import AdaBoostClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=AdaBoostClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.knn import KNNClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=KNNClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.svm import SVClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=SVClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.svm import NuSVClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=NuSVClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.sgd import SGDClassificacao
    otimizador = Otimizador(
        X=dados_previsores,
        y=dados_alvo,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=SGDClassificacao,
        criterio_evolucao='acuracia',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()'''

def executar_otimizacao_izi(n_geracoes, n_populacao, SEED=42,path_dataset_empreendimentos = 'datasets\\todos_absoluto.xlsx'):
    """Executa as evoluções definidas neste escopo para a Izi, por determinadas gerações e população."""

    # DATASET DOS EMPREENDIMENTOS
    # path_dataset_empreendimentos = 'datasets\\todos_absoluto.xlsx'
    
    dados_empreendimentos = Dados(
        caminho_arquivo_dados=path_dataset_empreendimentos,
        nome_dataset='empreendimentos',
        padronizar=True,
    )

    dados_independentes, dados_dependentes = dados_empreendimentos.get_conjuntos_completos_separados()

    from modelos.arvore_decisao import ArvoreDecisaoRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=ArvoreDecisaoRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False,
        log=True,
        verbose=True,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=ArvoreDecisaoRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False,
        log=True,
        verbose=True,
    )
    otimizador.evoluir()

    '''from modelos.rede_neural import RedeNeuralRegressao
    otimizador = Otimizador(
        y=dados_dependentes,
        X=dados_independentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=RedeNeuralRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=RedeNeuralRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.floresta_aleatoria import FlorestaAleatoriaRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=FlorestaAleatoriaRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=FlorestaAleatoriaRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.gradient_boosting import GradientBoostingRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=GradientBoostingRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=GradientBoostingRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.adaboost import AdaBoostRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=AdaBoostRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=AdaBoostRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.svm import NuSVRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=NuSVRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()
    
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=NuSVRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.sgd import SGDRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=SGDRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=SGDRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()'''

def executar_otimizacao_bru_3_0_regressao(n_geracoes, n_populacao, SEED=42, path_dataset_analisecredito = 'datasets\\sorted_data_sub.xlsx'):
    """Executa as evoluções definidas neste escopo para a Bru com regressao, por determinadas gerações e população."""

    # DATASET DOS EMPREENDIMENTOS
    # path_dataset_empreendimentos = 'datasets\\todos_absoluto.xlsx'
    
    dados_analise_credito = Dados(
        caminho_arquivo_dados=path_dataset_analisecredito,
        nome_dataset='analise_credito',
        padronizar=True,
    )

    dados_independentes, dados_dependentes = dados_analise_credito.get_conjuntos_completos_separados()

    from modelos.arvore_decisao import ArvoreDecisaoRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=ArvoreDecisaoRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False,
        log=True,
        verbose=False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=ArvoreDecisaoRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False,
        log=True,
        verbose=False,
    )
    otimizador.evoluir()

    """from modelos.rede_neural import RedeNeuralRegressao
    otimizador = Otimizador(
        y=dados_dependentes,
        X=dados_independentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=RedeNeuralRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False,
        log=True,
        verbose=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=RedeNeuralRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.floresta_aleatoria import FlorestaAleatoriaRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=FlorestaAleatoriaRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=FlorestaAleatoriaRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.gradient_boosting import GradientBoostingRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=GradientBoostingRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=GradientBoostingRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.adaboost import AdaBoostRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=AdaBoostRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=AdaBoostRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.svm import NuSVRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=NuSVRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()
    
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=NuSVRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    from modelos.sgd import SGDRegressao
    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=SGDRegressao,
        criterio_evolucao='r2',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X=dados_independentes,
        y=dados_dependentes,
        n_geracoes=n_geracoes,
        n_populacao=n_populacao,
        classe_modelo=SGDRegressao,
        criterio_evolucao='mae',
        random_state=SEED,
        por_validacao_cruzada=False
    )
    otimizador.evoluir()"""

def executar_otimizacao_bru_3_0_classificacao(n_geracoes, n_populacao, SEED, path_dataset_analisecredito):
    """Executa as evoluções definidas neste escopo para a Bru com regressao, por determinadas gerações e população."""
    dados_analise_credito = Dados(
        caminho_arquivo_dados=path_dataset_analisecredito,
        nome_dataset='analise_credito',
        padronizar=True,
    )

    dados_independentes, dados_dependentes = dados_analise_credito.get_conjuntos_completos_separados()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = AdaBoostClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = ArvoreDecisaoClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = FlorestaAleatoriaClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = GradientBoostingClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = KNNClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = NaiveBayesClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = RedeNeuralClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = RegressaoLogistica,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = SVClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()

    otimizador = Otimizador(
        X = dados_independentes,
        y = dados_dependentes,
        n_geracoes = n_geracoes,
        n_populacao = n_populacao,
        classe_modelo = SGDClassificacao,
        criterio_evolucao = 'acuracia',
        random_state = SEED,
        por_validacao_cruzada = True,
        log = True,
        verbose = False,
    )
    otimizador.evoluir()


# if(__name__ == '__main__'):
#     warnings.simplefilter(action = "ignore", category = RuntimeWarning)
# 
#     seed = 1
#     geracoes = 4 # Quantidade de geracoes para evoluir a populacao
#     populacao = 400 # Número de individuos
# 
#     executar_otimizacao_izi(n_geracoes=geracoes, n_populacao=populacao)
#     executar_otimizacao_bru(n_geracoes=geracoes, n_populacao=populacao, SEED = seed)
#     executar_otimizacao_bru_3_0_regressao(n_geracoes = geracoes, n_populacao = populacao, SEED = seed)
# 
    