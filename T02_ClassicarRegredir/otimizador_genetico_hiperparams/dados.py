from pandas import DataFrame, Series, read_excel
from numpy import ndarray
from sklearn.preprocessing import (
    StandardScaler,
    # MinMaxScaler
)


class Dados():
    """Representa o conjunto de dados que será utilizado para aplicar a evolução genética.
    Também possui funções de tratamento específico dos conjuntos de dados, padronização e filtragem.
    
    Parameters:
        caminho_arquivo_dados: path para o arquivo de dados.
        nome_dataset: nome do dataset que está sendo passado (identifica tratamentos específicos).
            'empreendimentos': dataset dos empreendimentos, exceto colunas de dados de município.
            'empreendimentos_todas_colunas':  dataset dos empreendimentos com todas as colunas.
            'analise_credito_classificacao': dataset com informações da BrU para classificacao.
            'analise_credito_regressao': dataset com informações da BrU para regressao.
        padronizar: se True aplica StandardScaler nos dados, não o faz caso contrário.
    """

    def __init__(
        self,
        caminho_arquivo_dados: str,
        nome_dataset: str = 'empreendimentos',
        padronizar: bool = False
    ) -> None:
        """Inicializa o objeto Dados útil para otimização de hiperparâmetros quando
            se tem um dataset preparado.

        - Em sua inicialização pode realizar algumas operações de tratamento dos dados.

        Args:
            caminho_arquivo_dados (str): path onde se encontra a base de dados utilizada.\n
            nome_dataset (str, optional): nome do conjunto de dados utilizado.
                - Default: 'empreendimentos'.\n
            padronizar (bool, optional): _description_. Default: False.
        """
        if nome_dataset == 'empreendimentos':
            colunas_a_remover = [
                'nome_empreendimento',
                'municipio',
                'capital_regional',
                'area_municipio_km2',
                'renda_media_municipio_sal_min',
                #'qtd_total_lotes',
                'qtd_lotes_disponivel',
                #'preco_lote_individual',
                #'area_lote_individual_m2',
                #'area_influencia_km2',
                #'raio_influencia_km',
                'dist_capital_regional_km',
                'tempo_ate_capital_regional_min',
                'dist_centro_municipio_km',
                'tempo_ate_centro_municipio_min',
                'inicio_venda',
                'data_atualizacao',
                #'meses_desde_inicio_venda',
                #'entrada',
                'maior_entrada_municipio',
                #'prazo',
                'maior_prazo_municipio',
                #'taxa_juros',
                'maior_taxa_juros_municipio',
                #'parcela',
                'maior_parcela_municipio',
                'eh_condominio_fechado',
                'eh_oferta_parcial',
                'pesquisou_inicio_venda',
                'consta_planilha_analise_viab_mercado',
                'consta_mymaps',
                'score_absoluto',
                #'curva_de_venda',
                'descricao',
                'centro_geolocalizacao',
                'coordenadas',
                'id_empreendimento',
            ]
            self.__get_filtragem_curva_de_venda(caminho_arquivo_dados, colunas_a_remover)
        elif nome_dataset == 'empreendimentos_todas_colunas':
            colunas_a_remover = [
                'nome_empreendimento',
                'municipio',
                'capital_regional',
                #'area_municipio_km2',
                #'renda_media_municipio_sal_min',
                #'qtd_total_lotes',
                'qtd_lotes_disponivel',
                #'preco_lote_individual',
                #'area_lote_individual_m2',
                #'area_influencia_km2',
                #'raio_influencia_km',
                #'dist_capital_regional_km',
                #'tempo_ate_capital_regional_min',
                #'dist_centro_municipio_km',
                #'tempo_ate_centro_municipio_min',
                'inicio_venda',
                'data_atualizacao',
                #'meses_desde_inicio_venda',
                #'entrada',
                #'maior_entrada_municipio',
                #'prazo',
                #'maior_prazo_municipio',
                #'taxa_juros',
                #'maior_taxa_juros_municipio',
                #'parcela',
                #'maior_parcela_municipio',
                'eh_condominio_fechado',
                'eh_oferta_parcial',
                'pesquisou_inicio_venda',
                'consta_planilha_analise_viab_mercado',
                'consta_mymaps',
                'score_absoluto',
                #'curva_de_venda',
                'descricao',
                'centro_geolocalizacao',
                'coordenadas',
                'id_empreendimento',
            ]
            self.__get_filtragem_curva_de_venda(caminho_arquivo_dados, colunas_a_remover)
        elif nome_dataset == 'analise_credito_classificacao':
            dados_brutos = read_excel(caminho_arquivo_dados)

            # classe_binaria_tier = {
            #     'S': 1,
            #     'A+': 1,
            #     'A': 1,
            #     'B+': 1,
            #     'B': 1,
            #     'C': 0,
            #     'D': 0,
            #     'D+': 0,
            #     'E': 0,
            #     'F': 0
            # }
            remap_binario_pos_neg = {'POS': 1, 'NEG': 0}
            dados_brutos['Tier'] = dados_brutos['Tier'].replace(remap_binario_pos_neg)

            self.param_prev = dados_brutos.drop(columns=['Tier'])
            self.param_alvo = dados_brutos['Tier']
        elif nome_dataset == 'analise_credito_regressao':
            dados_brutos = read_excel(caminho_arquivo_dados)

            n_tier = {
                'S': 0.5324,
                'A+': 0.4804,
                'A': 0.403,
                'B+': 0.3487,
                'B': 0.2084,
                'C': 0.0,
                'D+': -0.0014,
                'D': -0.2849,
                'E': -0.4772,
                'F': -0.9983,
            }

            dados_brutos['Tier'] = dados_brutos['Tier'].replace(n_tier)

            self.param_prev = dados_brutos.drop(columns=['Tier'])
            self.param_alvo = dados_brutos['Tier']

        if padronizar:
            self.param_prev = DataFrame(StandardScaler().fit_transform(
                DataFrame(data = self.param_prev, columns = self.param_prev.columns)),
                columns = self.param_prev.columns,
                index = self.param_prev.index
            )

    def __get_filtragem_curva_de_venda(
            self,
            caminho_arquivo_dados: str,
            colunas_a_remover: list[str]
        ) -> None:
        """Faz a filtragem de registros de empreendimentos de acordo com um limite para o
        valor de curva_de_venda estabelecido.
        
        - Utilizado somente para o dataset dos empreendimentos.
        - Não tem retorno, pois modifica os dados inplace.

        Parameters:
            caminho_arquivo_dados (str): path para o arquivo de dados.
            colunas_a_remover (list[str]): colunas a serem descartados do conjunto de dados.
        """
        limite_curva_venda = 31

        dados_brutos = read_excel(caminho_arquivo_dados)
        dados_brutos.drop(columns = colunas_a_remover, axis = 1, inplace = True)

        dados_brutos = dados_brutos.loc[dados_brutos['curva_de_venda'] < limite_curva_venda]

        self.param_prev = dados_brutos.drop(columns=['curva_de_venda'])
        self.param_alvo = dados_brutos['curva_de_venda']


    def get_conjuntos_completos_separados(self) -> tuple[ndarray, ndarray]:
        """Retorna os conjuntos de parâmetros independentes e o conjunto do parâmetro dependente.
        
        Returns:
            numpy.ndarray, numpy.ndarray: variáveis independentes ou parâmetros previsores
                e variável dependente ou parâmetro alvo, respectivamente.
        """
        return self.param_prev.values, self.param_alvo.values
