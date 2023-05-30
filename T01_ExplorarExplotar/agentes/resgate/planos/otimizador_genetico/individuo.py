from random import choice
from copy import deepcopy


class Individuo:
    def __init__(
            self,
            caminhos: dict[str, dict[str, list[tuple[str, int]]]],
            tempo_restante: float,
            max_subcaminhos: int
        ) -> None:
        self.tempo_restante: float = tempo_restante
        self.max_subcaminhos: int = max_subcaminhos
        self.qtd_subcaminhos: int = 0
        self.pontuacao: int = 0
        self.caminhos = caminhos
        self.caminhos_possiveis = []
        self.ja_visitados = []

        # GENES E INFORMAÇÕES CARACTERÍSTICAS
        self.genes: dict[str, list[tuple[str, str]] | int] = {
            'trajeto': [], # [(origem, destino1), (destino1, destino2), ..., (destino2, origem)]
            'peso_gravidade': 0,
            'peso_custo': 0
        }

        self.parametros_possiveis = {
            'peso_custo': [
                0.05, 0.1,
                0.15, 0.2,
                0.25, 0.3,
                0.35
            ],
            'peso_gravidade': [
                0.95, 0.9,
                0.85, 0.8,
                0.75, 0.7,
                0.65
            ]
        }

    def __str__(self) -> str:
        """Retorna o objeto representado por meio de uma saída de texto.

        Returns:
            str: indivíduo no formato para ser representado na saída de texto.
        """
        return f"Tamanho: {self.qtd_subcaminhos}\nGenes: {str(self.genes)}"

    def set_caracteristicas(self, caracteristicas: dict[str, float]) -> None:
        """Define as características de peso do indivíduo.

        Args:
            caracteristicas (dict[str, float]): dicionário com os pesos para função de fitness.
        """
        self.genes['peso_gravidade'] = caracteristicas['peso_gravidade']
        self.genes['peso_custo'] = caracteristicas['peso_custo']

    def set_trajeto(self, trajeto: list[tuple[str, str]]) -> None:
        """Atribui a sequência de sub-caminhos para um indivíduo.

        Args:
            trajeto (list[tuple[str, str]]): lista de sub-caminhos a ser atribuída.
        """
        self.genes['trajeto'] = trajeto

    def avalia_individuo(self, vitimas: dict[str, list[str]]):
        """Efetua a avaliação de um indivíduo, (aplica sua função de fitness).

        Args:
            vitimas (dict[str, list[str]]): as vítimas conhecidas no ambiente.
        """
        pontuacoes = []
        ordem_salvamento = self.qtd_subcaminhos + 1
        vitimas_salvas = {}

        for origem, destino in self.genes['trajeto']:
            if destino in vitimas and destino not in vitimas_salvas:
                gravidade = int(vitimas[destino][-1])
                gravidade_normalizada = ((4 - gravidade) + 1) * ordem_salvamento
                vitimas_salvas[destino] = deepcopy(vitimas[destino])
                self.tempo_restante -= 1
            else:
                gravidade_normalizada = 0
            ordem_salvamento -= 1

            custo_subcaminho = self.__obtem_custo_caminho(origem, destino)
            pontuacao_subcaminho = (
                (self.genes['peso_gravidade'] * gravidade_normalizada)
                / (self.genes['peso_custo'] * custo_subcaminho)
            )

            pontuacoes.append(pontuacao_subcaminho)

        self.pontuacao = sum(pontuacoes)

        cont = 0
        penalidade = 0
        for ordem, percurso in enumerate(self.genes['trajeto']):
            if percurso[0] == ['0:0'] or percurso[1] == ['0:0']:
                penalidade += self.qtd_subcaminhos - ordem + 1
                cont += 1

        if cont > 2:
            self.pontuacao -= penalidade

    def gera_individuo_aleatorio(self):
        """Gera indivíduos aleatórios de acordo com algumas condições.

        1. A origem do primeiro sub-caminho de todo individuo deve ser a posição da base
            dos agentes.
        2. O destino do  ́ultimo sub-caminho de todo individuo deve ser a posição da base
            dos agentes.
        3. A posição conhecida de uma vítima deve aparecer uma  ́unica vez como destino e
            outra vez como origem em sub-caminhos distintos.
        4. A origem do sub-caminho seguinte deve ser a posição da vítima que foi o destino
            do  ́ultimo sub-caminho.
        5. O custo combinado de todos os sub-caminhos realizados deve ser inferior ao
            tempo disponível.
        """
        chave_posicao_atual = '0:0'

        while self.__tem_caminho_possivel_no_tempo(chave_posicao_atual):
            if len(self.genes['trajeto']) >= self.max_subcaminhos:
                break

            if len(self.caminhos_possiveis) > 1:
                destino_escolhido = choice(self.caminhos_possiveis)
                if destino_escolhido == '0:0':
                    for caminhozinho in self.caminhos_possiveis:
                        if (
                            caminhozinho != "0:0"
                            and self.__eh_possivel_ir_e_voltar(chave_posicao_atual, caminhozinho)
                        ):
                            if '0:0' in self.caminhos_possiveis:
                                self.caminhos_possiveis.remove('0:0')
                            while destino_escolhido == '0:0':
                                destino_escolhido = choice(self.caminhos_possiveis)
                            break

                self.genes['trajeto'].append([chave_posicao_atual, destino_escolhido])
                self.tempo_restante -= self.__obtem_custo_caminho(
                    chave_origem = chave_posicao_atual,
                    chave_destino = destino_escolhido
                )
                chave_posicao_atual = destino_escolhido
                self.qtd_subcaminhos += 1
                if chave_posicao_atual != '0:0':
                    self.ja_visitados.append(destino_escolhido)
            elif '0:0' in self.caminhos_possiveis:
                if (self.__eh_possivel_ir_e_voltar(
                        origem = chave_posicao_atual,
                        destino = '0:0'
                    )
                ):
                    self.genes['trajeto'].append([chave_posicao_atual, "0:0"])
                    self.tempo_restante -= self.__obtem_custo_caminho(
                        chave_origem = chave_posicao_atual,
                        chave_destino = '0:0'
                    )
                    self.qtd_subcaminhos += 1
            else:
                break

            self.caminhos_possiveis = []
        self.genes['peso_custo'] = choice([
            0.05, 0.1,
            0.15, 0.2,
            0.25, 0.3,
            0.35
        ])
        self.genes['peso_gravidade'] = 1 - self.genes['peso_custo']

    def __tem_caminho_possivel_no_tempo(self, chave_origem: str) -> bool:
        """Retorna se ainda há algum sub-caminho possível para realizar.

        Leva em consideração a ida até o destino e a volta até a base dos agentes.

        Args:
            chave_origem (str): origem de partida do sub-caminho.

        Returns:
            bool: True se é possível ir até o sub-caminho e voltar até a base dos agentes,
                False caso contrário.
        """
        for destino in self.caminhos[chave_origem]:
            if (
                self.__eh_possivel_ir_e_voltar(chave_origem, destino)
                and destino not in self.ja_visitados
            ):
                self.caminhos_possiveis.append(destino)

        if self.caminhos_possiveis:
            return True
        return False

    def __eh_possivel_ir_e_voltar(self, origem: str, destino: str) -> bool:
        """Retorna se, partindo de determinada origem, é possível chegar até o destino
            e, deste destino, retornar até a base dos agentes.

        Args:
            origem (str): origem do sub-caminho.
            destino (str): destino do sub-caminho.

        Returns:
            bool: True se é possível ir até o destino e voltar até a base dos agentes,
                False caso contrário.
        """
        if destino == origem:
            return True
        custo_ida = self.__obtem_custo_caminho(origem, destino)
        custo_volta_base = self.__obtem_custo_caminho(destino, '0:0')
        if (custo_ida + custo_volta_base) <= self.tempo_restante:
            return True
        return False

    def __obtem_custo_caminho(self, chave_origem: str, chave_destino: str) -> float:
        """Retorna o custo de tempo para realizar o sub-caminho pretendido.

        Args:
            chave_origem (str): origem do sub-caminho.
            chave_destino (str): destino do sub-caminho.

        Returns:
            float: custo de tempo para realizar o sub-caminho.
        """
        if chave_origem == chave_destino:
            return 0

        custo = 0
        percurso = self.caminhos[chave_origem][chave_destino]

        custo = sum(posicao[1] for posicao in percurso)

        return custo
