import heapq
import math

class Node:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state = state          # Tupla com os pacientes restantes (id, [p_b, p_m, p_a], tempo)
        self.parent = parent
        self.action = action        # ID do paciente que foi atendido nesta ação
        self.path_cost = path_cost  # g(n): Risco acumulado total até aqui

    def __lt__(self, other):
        # Utilizado para comparar nós diretamente
        return self.path_cost < other.path_cost

    def __repr__(self):
        return f"<Node {len(self.state)} pacientes na fila>"

class Problem:
    def __init__(self, initial_state, goal_state=None):
        self.initial_state = initial_state
        self.goal_state = goal_state

    def actions(self, state):
        raise NotImplementedError

    def result(self, state, action):
        raise NotImplementedError

    def is_goal(self, state):
        # O objetivo é atingido quando a fila está vazia
        return len(state) == 0

    def action_cost(self, s, action, s1):
        raise NotImplementedError


class TriageProblem(Problem):
    def __init__(self, initial_state):
        # O estado final é uma tupla vazia ()
        super().__init__(initial_state, goal_state=())
        # Mapeamento de Tempo de Atendimento por Gravidade
        self.tempo_atendimento = {
            'alta': 20,
            'media': 15,
            'baixa': 10
        }

    def obter_categoria_gravidade(self, probs):
        """Identifica a classe predominante do vetor [p_baixa, p_media, p_alta]"""
        prob_max = max(probs)
        idx_max = probs.index(prob_max)
        
        if idx_max == 2:
            return 'alta'
        elif idx_max == 1:
            return 'media'
        else:
            return 'baixa'

    def actions(self, state):
        """As ações possíveis são os pacientes disponíveis para atendimento"""
        # Retorna os ids de todos os pacientes que ainda estão no estado atual
        return [paciente[0] for paciente in state]

    def result(self, state, action):
        """Gera o novo estado removendo o paciente pelo ID e avançando o tempo dos demais."""
        # Encontra o paciente que será atendido (onde o ID bate com a action)
        paciente_atendido = next(p for p in state if p[0] == action)
        # Vetor de probabilidades: [p_b, p_m, p_a]
        vetor_probs = paciente_atendido[1]
        # Define a gravidade do paciente e a duração do atendimento
        gravidade_cat = self.obter_categoria_gravidade(vetor_probs)
        duracao_atendimento = self.tempo_atendimento[gravidade_cat]
        
        novo_estado = []
        for p in state:
            if p[0] != action:  # Se não for o ID do paciente atendido
                novo_tempo = p[2] + duracao_atendimento
                novo_estado.append((p[0], p[1], novo_tempo))
                
        return tuple(novo_estado), duracao_atendimento

    def action_cost(self, s, action, s1, duracao_atendimento):
        """Calcula o custo de deterioração com base no ID do paciente atendido."""
        custo_transicao = 0
        for p in s:
            if p[0] != action:
                # usamos apenas a P(gravidade_alta)
                p_alta = p[1][2] 
                tempo_final = p[2] + duracao_atendimento
                
                custo_transicao += p_alta * tempo_final
                
        return custo_transicao

# Reconstroi a solução
def solution(node):
    actions = []
    total_cost = node.path_cost
    curr = node
    while curr.parent is not None:
        actions.append(curr.action)
        curr = curr.parent
    return actions[::-1], total_cost

# Expande os nós
def expand(problem, node):
    s = node.state
    for action in problem.actions(s):
        # O result devolve duas coisas: o próximo estado e a duração da consulta
        s1, duracao = problem.result(s, action)
        
        # Passamos a duracao diretamente para o custo da ação
        cost = node.path_cost + problem.action_cost(s, action, s1, duracao)
        yield Node(state=s1, parent=node, action=action, path_cost=cost)

# Heurística

def h_triage(node):
    """Calcula a soma dos riscos atuais de todos os pacientes restantes"""
    total_h = 0
    for paciente in node.state:
        p_alta = paciente[1][2]
        tempo_espera = paciente[2]
        # risco = P(gravidade_alta) * tempo_esperando
        total_h += p_alta * tempo_espera
    return total_h


# Algoritmo A*

def a_star_search(problem, h_func):
    node = Node(problem.initial_state)
    frontier = []
    
    # Armazena na fila de prioridades uma tupla: (f(n), id_unico, node)
    counter = 0  # Evita erro de comparação entre instâncias de Node caso f(n) empate
    heapq.heappush(frontier, (node.path_cost + h_func(node), counter, node))
    reached = {problem.initial_state: node}

    while frontier:
        
        priority, _, node = heapq.heappop(frontier)
        if problem.is_goal(node.state):
            return solution(node)

        for child in expand(problem, node):
            s = child.state
            if s not in reached or child.path_cost < reached[s].path_cost:
                reached[s] = child
                counter += 1
                heapq.heappush(frontier, (child.path_cost + h_func(child), counter, child))
                
    return None, float('inf')

def _simular_fila_base(problem, chave_ordenacao):
    """
    Função genérica que roda a simulação do pronto-socorro.
    Reordena a fila a cada ciclo com base na regra fornecida.
    """
    # Cada elemento p é: (id, [p_baixa, p_media, p_alta], tempo_espera)
    fila = list(problem.initial_state)
    custo_total_acumulado = 0

    while fila:
        # Ordena dinamicamente a cada ciclo antes de chamar o próximo
        fila.sort(key=chave_ordenacao, reverse=True)
        
        # Remove o primeiro da fila conforme a ordenação atual
        atendido = fila.pop(0)
        
        # Calcula a gravidade e o tempo de atendimento do paciente atual
        vetor_probs = atendido[1]
        gravidade_cat = problem.obter_categoria_gravidade(vetor_probs)
        duracao_consulta = problem.tempo_atendimento[gravidade_cat]
        
        # Atualiza o tempo de quem ficou e calcula o custo dessa transição
        nova_fila = []
        for p in fila:
            id_paciente = p[0]
            probs_paciente = p[1]
            p_alta = p[1][2] # P(Gravidade Alta) para a fórmula de risco
            
            # O tempo de espera aumenta com a duração da consulta atual
            novo_tempo_espera = p[2] + duracao_consulta
            
            # Calcula o risco acumulado com base no novo tempo
            custo_total_acumulado += p_alta * novo_tempo_espera
            
            # Guarda o paciente com o tempo atualizado para as próximas rodadas
            nova_fila.append((id_paciente, probs_paciente, novo_tempo_espera))
            
        # Atualiza a fila com os tempos avançados para o próximo ciclo
        fila = nova_fila

    return custo_total_acumulado

def FIFO(problem):
    """Atende na ordem de chegada (maior tempo de espera acumulado primeiro)."""
    return _simular_fila_base(problem, chave_ordenacao=lambda p: p[2])


def greedy(problem):
    """Atende sempre o de maior probabilidade de gravidade alta."""
    return _simular_fila_base(problem, chave_ordenacao=lambda p: p[1][2])