from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def inicializar_rede():
    """
    Define a estrutura correta da Rede Bayesiana (Sintomas -> Gravidade)
    e configura as CPTs com base nos estados definidos.
    """
    # Estrutura formal exigida pelo enunciado (Página 1)
    modelo = DiscreteBayesianNetwork([
        ('Febre', 'Gravidade'),
        ('Saturacao', 'Gravidade'),
        ('PressaoArt', 'Gravidade'),
        ('FreqCard', 'Gravidade'),
        ('NivelDor', 'Gravidade')
    ])

    # Definição dos nomes de estados para consistência na inferência
    FEBRE_STATES = ['Sim', 'Nao']
    SATURACAO_STATES = ['Normal', 'Reduzida', 'Critica']
    PRESART_STATES = ['Normal', 'Baixa', 'Alta']
    FREQCARD_STATES = ['Normal', 'Baixa', 'Alta']
    DOR_STATES = ['Leve', 'Moderada', 'Intensa']
    GRAVIDADE_STATES = ['Alta', 'Media', 'Baixa']

    # Distribuições a priori dos sintomas (Probabilidades Marginais)
    cpd_febre = TabularCPD('Febre', 2, [[0.4], [0.6]], state_names={'Febre': FEBRE_STATES})
    cpd_saturacao = TabularCPD('Saturacao', 3, [[0.7], [0.2], [0.1]], state_names={'Saturacao': SATURACAO_STATES})
    cpd_pressao = TabularCPD('PressaoArt', 3, [[0.7], [0.2], [0.1]], state_names={'PressaoArt': PRESART_STATES})
    cpd_freq = TabularCPD('FreqCard', 3, [[0.7], [0.15], [0.15]], state_names={'FreqCard': FREQCARD_STATES})
    cpd_dor = TabularCPD('NivelDor', 3, [[0.5], [0.3], [0.2]], state_names={'NivelDor': DOR_STATES})

    # CPT Central: Gravidade condicionada a todos os 5 sintomas simultaneamente
    # Total de colunas combinadas = 2 (Febre) * 3 (Sat) * 3 (Pressao) * 3 (Freq) * 3 (Dor) = 162 colunas
    total_combinacoes = 162
    
    # Preenchimento em lote com valores base plausíveis
    p_alta = [0.15] * total_combinacoes
    p_media = [0.50] * total_combinacoes
    p_baixa = [0.35] * total_combinacoes

    # Mapeamento dinâmico para emular regras críticas do Protocolo de Manchester
    # (Ex: Saturação Crítica [índice 2] eleva drasticamente a probabilidade de Gravidade Alta)
    coluna = 0
    for f in range(2):
        for s in range(3):
            for p in range(3):
                for fc in range(3):
                    for d in range(3):
                        if s == 2:  # Saturação Crítica (< 90%)
                            p_alta[coluna] = 0.85
                            p_media[coluna] = 0.10
                            p_baixa[coluna] = 0.05
                        elif s == 1 or p == 1 or fc == 2:  # Sinais vitais moderadamente alterados
                            p_alta[coluna] = 0.30
                            p_media[coluna] = 0.60
                            p_baixa[coluna] = 0.10
                        coluna += 1

    cpd_gravidade = TabularCPD(
        variable='Gravidade',
        variable_card=3,
        values=[p_alta, p_media, p_baixa],
        evidence=['Febre', 'Saturacao', 'PressaoArt', 'FreqCard', 'NivelDor'],
        evidence_card=[2, 3, 3, 3, 3],
        state_names={'Gravidade': GRAVIDADE_STATES, 'Febre': FEBRE_STATES, 
                     'Saturacao': SATURACAO_STATES, 'PressaoArt': PRESART_STATES, 
                     'FreqCard': FREQCARD_STATES, 'NivelDor': DOR_STATES}
    )

    modelo.add_cpds(cpd_febre, cpd_saturacao, cpd_pressao, cpd_freq, cpd_dor, cpd_gravidade)
    modelo.check_model()
    return VariableElimination(modelo)

# Instancia o motor de inferência global
engine_inferencia = inicializar_rede()

def calcular_probabilidade_gravidade(sintomas):
    """
    Recebe as evidências e retorna o vetor [P(Baixa), P(Media), P(Alta)].
    Garante o mapeamento correto dos índices conforme esperado pelo Módulo 2.
    """
    resultado = engine_inferencia.query(variables=['Gravidade'], evidence=sintomas, show_progress=False)
    
    # Mapeamento do pgmpy: índice 0=Alta, 1=Media, 2=Baixa.
    # Invertemos o retorno para respeitar o padrão do busca_triage: [P_Baixa, P_Media, P_Alta]
    p_alta = float(resultado.values[0])
    p_media = float(resultado.values[1])
    p_baixa = float(resultado.values[2])
    
    return [p_baixa, p_media, p_alta]