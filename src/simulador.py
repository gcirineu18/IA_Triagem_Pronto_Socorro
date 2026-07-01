import random
from rede_bayesiana import calcular_probabilidade_gravidade
from busca_triage import TriageProblem, a_star_search, h_triage, FIFO, greedy

def gerar_cenario_medio():
    """Gera uma base controlada de 20 pacientes com sintomas diversos."""
    random.seed(42)  # Semente fixa para garantir testes idênticos entre os algoritmos
    
    estados_sintomas = {
        'Febre': ['Sim', 'Nao'],
        'Saturacao': ['Normal', 'Reduzida', 'Critica'],
        'PressaoArt': ['Normal', 'Baixa', 'Alta'],
        'FreqCard': ['Normal', 'Baixa', 'Alta'],
        'NivelDor': ['Leve', 'Moderada', 'Intensa']
    }
    
    pacientes_brutos = []
    for i in range(1, 21):
        pacientes_brutos.append({
            'id': f'P{i:03d}',
            'sintomas': {k: random.choice(v) for k, v in estados_sintomas.items()},
            'tempo_espera': random.randint(0, 30)  # Tempo de espera inicial na recepção
        })
    return pacientes_brutos

def rodar_simulacao():
    print("=" * 65)
    print("SISTEMA INTEGRADOR DE TRIAGEM HOSPITALAR — EXPERIMENTOS")
    print("=" * 65)
    
    dados_recepcao = gerar_cenario_medio()
    fila_preparada = []
        
    # Módulo de Integração: Alimentando Módulo 2 com dados do Módulo 1
    for paciente in dados_recepcao:
        vetor_probabilidades = calcular_probabilidade_gravidade(paciente['sintomas'])
        
        # CORREÇÃO CRUCIAL: Converta a lista de probabilidades em uma TUPLA imutável
        vetor_probs_imutavel = tuple(vetor_probabilidades)
        
        # Monta a estrutura rigorosa usando a tupla convertida:
        fila_preparada.append((paciente['id'], vetor_probs_imutavel, paciente['tempo_espera']))
        
    estado_inicial_imutavel = tuple(fila_preparada)

    # Instancia instâncias limpas do problema para cada algoritmo
    problema_astar = TriageProblem(estado_inicial_imutavel)
    problema_fifo = TriageProblem(estado_inicial_imutavel)
    problema_greedy = TriageProblem(estado_inicial_imutavel)
    
    print(f"[*] Base de {len(dados_recepcao)} pacientes carregada e processada via pgmpy.")
    print("-" * 65)
    
    # Execução das estratégias comparativas
    custo_fifo = FIFO(problema_fifo)
    print(f"[Estratégia FIFO]   Risco Total Acumulado: {custo_fifo:.2f}")
    
    custo_greedy = greedy(problema_greedy)
    print(f"[Estratégia Gulosa] Risco Total Acumulado: {custo_greedy:.2f}")
    
    ordem_astar, custo_astar = a_star_search(problema_astar, h_triage)
    print(f"[Algoritmo A*]     Risco Total Acumulado: {custo_astar:.2f}")
    print("-" * 65)
    print(f" Ordem de chamada ótima definida pelo A*:\n {ordem_astar}")
    print("=" * 65)

if __name__ == "__main__":
    rodar_simulacao()