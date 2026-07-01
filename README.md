# Sistema Inteligente de Triagem em Pronto-Socorro
---

## 📋 Sobre o Projeto
Este projeto propõe uma solução inteligente para mitigar o grave problema de superlotação e lentidão em pronto-socorros brasileiros. O sistema é composto por dois módulos principais e interdependentes que visam otimizar o fluxo de atendimento médico de urgência:

1. **Módulo 1 — Rede Bayesiana:** Estima e diagnostica a probabilidade de gravidade do quadro clínico do paciente com base em sintomas e sinais vitais (Febre, Saturação de O₂, Pressão Arterial, Frequência Cardíaca e Nível de Dor) utilizando inferência probabilística com a biblioteca `pgmpy`.
2. **Módulo 2 — Algoritmo de Busca ($A^*$):** Resolve o problema de ordenação dinâmica da fila de atendimento. O algoritmo encontra a sequência ótima de chamadas que minimiza o risco de deterioração acumulado total de todos os pacientes que aguardam na sala de espera.

A fusão central do sistema reside no fato de que a probabilidade de gravidade alta calculada pela Rede Bayesiana atua diretamente como o fator de aceleração temporal (peso) na função de custo e na heurística do $A^*$. Sem a rede, o algoritmo de busca não teria dados clínicos para estimar o risco de cada paciente.

---

## 📂 Estrutura do Repositório
O projeto está organizado sob a seguinte arquitetura modular e limpa:

```text
triagem-hospitalar-ia/
│
├── notebooks/
│   └── rascunho_rede_bayesiana.ipynb  # Protótipo inicial de CPTs e testes gráficos
│
├── src/
│   ├── __init__.py                    # Inicializador de pacote Python (vazio)
│   ├── rede_bayesiana.py              # Definição e inferência com pgmpy (Módulo 1)
│   ├── busca_triage.py                # Algoritmos A*, FIFO e Guloso (Módulo 2)
│   └── simulador.py                   # Integração e orquestração dos experimentos de teste
│
├── requirements.txt                   # Dependências externas do projeto
└── README.md                          # Instruções gerais de uso

```

---

## 🛠️ Requisitos e Instalação

O projeto foi desenvolvido em Python 3. Certifique-se de possuir o gerenciador de pacotes `pip` instalado antes de prosseguir com os passos abaixo:

1. **Clone o repositório para a sua máquina local:**
```bash
git clone [https://github.com/SEU_USUARIO/triagem-hospitalar-ia.git](https://github.com/SEU_USUARIO/triagem-hospitalar-ia.git)
cd triagem-hospitalar-ia

```


2. **Instale todas as dependências do projeto de forma automatizada:**
```bash
pip install -r requirements.txt

```



*Nota: Os pacotes instalados incluem `pgmpy`, `networkx`, `numpy`, `pandas` e `matplotlib`.*

---

## 🚀 Como Executar o Simulador

Para rodar o cenário padrão de testes contendo os experimentos comparativos exigidos no enunciado do trabalho (Cenário de Médio Porte contendo uma fila de 20 pacientes), execute o script principal de integração:

```bash
python src/simulador.py

```

### O que o simulador executa por baixo dos panos?

1. **Geração de Casos:** Cria uma fila de entrada contendo os sinais clínicos estruturados e os tempos iniciais de chegada de 20 pacientes na recepção.
2. **Inferência Probabilística (Módulo 1):** Submete os sintomas de cada paciente à Rede Bayesiana (respeitando a topologia correta de Sintomas $\rightarrow$ Gravidade) para inferir a matriz de criticidade do Protocolo de Manchester.
3. **Mapeamento de Estados:** Organiza esses dados no formato de tuplas imutáveis `(id, [p_baixa, p_media, p_alta], tempo_espera)` exigido pelo Módulo 2.
4. **Confronto de Estratégias:** Dispara e compara de forma equivalente o custo de risco total sofrido sob as estratégias:
* **FIFO:** Atende estritamente por ordem de chegada, ignorando a gravidade.
* **Gulosa:** Atende sempre quem tem maior $P(\text{gravidade\_alta})$, ignorando o relógio.
* **A*:** Minimiza o risco acumulado total ponderando dinamicamente gravidade e tempo através de uma função heurística admissível.



Ao final, o terminal exibirá a auditoria de custos comprovando matematicamente a otimalidade e a redução de danos gerada pelo algoritmo $A^*$.

---

## 👥 Componentes do Grupo

* **Integrante 1 (Letícia Frota)** — 552539
* **Integrante 2 (Nome do Colega)** — RMxxxx
* **Integrante 3 (Nome do Colega)** — RMxxxx
* **Integrante 4 (Nome do Colega)** — RMxxxx

```

```