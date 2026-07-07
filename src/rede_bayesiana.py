from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

def inicializar_rede():

    modelo_diagnostico_gravidade = DiscreteBayesianNetwork([
        ('Gravidade', 'Febre'),
        ('Gravidade', 'Saturacao'),
        ('Gravidade', 'PressaoArt'),
        ('Gravidade', 'FreqCard'),
        ('Gravidade', 'NivelDor'),
    ])
    SATURACAO_STATES = ['Normal', 'Reduzida', 'Critica']
    GRAVIDADE_STATES = ['Alta', 'Media', 'Baixa']
    FEBRE_STATES = ['Sim', 'Nao']
    PRESART_STATES = ['Normal', 'Baixa', 'Alta']
    FREQCARD_STATES = ['Normal', 'Baixa', 'Alta']
    DOR_STATES = ['Leve', 'Moderada', 'Intensa']



    # Passo 2: Definir as Tabelas de Probabilidade Condicional (CPTs)
    # CPT para Gravidade (probabilidade a priori)
    # P(Gravidade=Baixa) = 0.325 (32,5% dos casos registrados
    # na referência são de baixa gravidade )
    cpd_gravidade = TabularCPD(
        variable="Gravidade",
        variable_card=3,
        values=[
                [0.325], # P(Gravidade=Baixa)
                [0.491], # P(Gravidade=Média)
                [0.184]  # P(Gravidade=Alta)
                ],
        state_names={
        "Gravidade": GRAVIDADE_STATES
    }
    )

    # CPT para a Saturação de O2 dada a Gravidade 
    cpd_saturacao = TabularCPD(
        variable="Saturacao",
        variable_card=3,
        values=[
            [0.10, 0.45, 0.85],  # Normal
            [0.2, 0.5, 0.12],  # Reduzida
            [0.7, 0.05, 0.03],  # Critica
        ],
        evidence=["Gravidade"],
        evidence_card=[3],
        state_names={
            "Gravidade": GRAVIDADE_STATES,
            "Saturacao": SATURACAO_STATES
        }
    )

    # CPT para a Febre dada a Gravidade
    cpd_febre = TabularCPD(
        variable="Febre",
        variable_card=2,
        values=[
            [0.60, 0.35, 0.10],  # Sim
            [0.40, 0.65, 0.90],  # Nao
        ],
        evidence=["Gravidade"],
        evidence_card=[3],
        state_names={
            "Gravidade": GRAVIDADE_STATES,
            "Febre": FEBRE_STATES
        }
    )

    # CPT para a Pressão Arterial dada a Gravidade
    cpd_pressaoArt = TabularCPD(
        variable="PressaoArt",
        variable_card=3,
        values=[
            [0.04, 0.35, 0.38],  # Normal
            [0.21, 0.45, 0.50],  # Baixa
            [0.75, 0.20, 0.12],  # Alta
        ],
        evidence=["Gravidade"],
        evidence_card=[3],
        state_names={
            "PressaoArt": PRESART_STATES ,
            "Gravidade": GRAVIDADE_STATES
        }
    )

    # CPT para a Frequência Cardíaca dada a Gravidade 
    cpd_freqCard = TabularCPD(
        variable="FreqCard",
        variable_card=3,
        values=[
            [0.04, 0.35, 0.38],  # Normal
            [0.21, 0.45, 0.50],  # Baixa
            [0.75, 0.20, 0.12],  # Alta
        ],
        evidence=["Gravidade"],
        evidence_card=[3],
        state_names={
            "Gravidade": GRAVIDADE_STATES,
            "FreqCard": FREQCARD_STATES
        }
    )

    # CPT para Nível da dor dada a Gravidade
    cpd_nivelDor = TabularCPD(
        variable="NivelDor",
        variable_card=3,
        values=[
            [0.03, 0.10, 0.55],  # Leve
            [0.17, 0.6, 0.3],  # Moderada
            [0.8, 0.3, 0.15],  # Intensa
        ],
        evidence=["Gravidade"],
        evidence_card=[3],
        state_names={
            "Gravidade": GRAVIDADE_STATES,
            "NivelDor": DOR_STATES
        }
    )

    modelo_diagnostico_gravidade.add_cpds(cpd_gravidade, cpd_saturacao, cpd_febre, cpd_pressaoArt, cpd_freqCard, cpd_nivelDor)

    modelo_diagnostico_gravidade.check_model()
    return VariableElimination(modelo_diagnostico_gravidade)

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