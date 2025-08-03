
"""
=============================================================================
RESOLVEDOR DE SISTEMAS LINEARES 3D - APLICAÇÃO STREAMLIT
=============================================================================

Aplicação interativa para resolução de sistemas de equações lineares 3x3
com visualização 3D passo a passo do processo de eliminação gaussiana.

Autor: Autor: Izaac Soares, Jeferson Danilo
Data: 03/08/2025
Versão: 1.0

Funcionalidades:
- Entrada interativa de coeficientes
- Resolução passo a passo via eliminação gaussiana
- Visualização 3D dos planos geométricos
- Formatação LaTeX das matrizes
- Interface responsiva e intuitiva

Dependências:
- streamlit
- numpy
- pandas
- plotly
- linear_solver (módulo personalizado)
"""

# =============================================================================
# IMPORTAÇÕES
# =============================================================================

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import io
import sys
import re
import itertools
from contextlib import redirect_stdout

# Importações do módulo personalizado
from linear_solver import (
    validar_matriz_usuario, 
    resolver_sistema_linear_passo_a_passo,
    plotar_planos_3d, 
    PLOT_CONFIG
)

# =============================================================================
# CONFIGURAÇÃO DA PÁGINA
# =============================================================================

st.set_page_config(
    page_title="Resolvedor de Sistemas Lineares 3D",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# CABEÇALHO PRINCIPAL
# =============================================================================

st.title("🔢 Resolvedor de Sistemas Lineares 3x3")
st.markdown("""
### Uma aplicação interativa para resolver sistemas de equações lineares com visualização 3D passo a passo

Esta ferramenta permite resolver sistemas de equações lineares 3x3 através do método de eliminação gaussiana,
visualizando cada etapa do processo tanto algebricamente quanto geometricamente.
""")

# =============================================================================
# FUNÇÕES DE FORMATAÇÃO E UTILITÁRIOS
# =============================================================================

def formatar_matriz_latex(matriz, highlight_row_index=None):
    """
    Converte uma matriz numérica para representação LaTeX formatada.
    
    Esta função cria uma matriz aumentada em notação LaTeX, adequada para
    sistemas de equações lineares, com separação visual entre coeficientes
    e termos independentes.
    
    Args:
        matriz (np.array or list): Matriz a ser formatada (deve ter 4 colunas)
        highlight_row_index (int, optional): Índice da linha a destacar
        
    Returns:
        str: String formatada em LaTeX para renderização matemática
        
    Exemplo:
        >>> matriz = [[1, 2, 3, 4], [5, 6, 7, 8]]
        >>> print(formatar_matriz_latex(matriz))
        $$\left(\begin{array}{ccc|c} 1 & 2 & 3 & 4 \\ 5 & 6 & 7 & 8 \end{array}\right)$$
    """
    # Validação de entrada
    if matriz is None or len(matriz) == 0 or len(matriz[0]) == 0:
        return ""
    
    # Conversão para lista se necessário
    matriz_list = matriz.tolist() if isinstance(matriz, np.ndarray) else matriz
    num_rows = len(matriz_list)
    num_cols = len(matriz_list[0])
    
    # Configuração do alinhamento (separador vertical antes da última coluna)
    alignment = 'c' * (num_cols - 1) + '|c'

    # Formatação de cada linha
    rows_latex = []
    for i, row in enumerate(matriz_list):
        formatted_elements = []
        for elem in row:
            # Conversão e formatação numérica
            val = float(elem)
            if np.isclose(val, int(val)):
                formatted_elements.append(str(int(val)))
            else:
                formatted_elements.append(f'{val:.1f}')
        
        row_str = ' & '.join(formatted_elements)
        rows_latex.append(row_str)
    
    # Montagem da matriz LaTeX
    matrix_body = ' \\\\ '.join(rows_latex)
    latex_string = f'$$\\left(\\begin{{array}}{{{alignment}}} {matrix_body} \\end{{array}}\\right)$$'
    
    return latex_string


def parse_step_operation(step_name):
    """
    Analisa a descrição textual de uma operação matricial para extrair informações estruturadas.
    
    Esta função usa expressões regulares para identificar diferentes tipos de operações
    de eliminação gaussiana e gerar descrições LaTeX apropriadas.
    
    Args:
        step_name (str): Descrição da operação (ex: 'L2 <- L2 - 1.0 * L1')
        
    Returns:
        tuple: (linha_alvo, linha_fonte, desc_curta, desc_longa)
               Retorna (None, None, None, None) se não conseguir analisar
               
    Tipos de operações suportadas:
        - Combinação linear: L_i <- L_i + c * L_j
        - Escalonamento: L_i <- c * L_i  
        - Permutação: L_i <-> L_j
    """
    # Padrão para operações de combinação linear (L_i <- L_i ± c * L_j)
    match = re.search(r'L(\d+) <- L\d+ ([+-]) ([\d\.]+) \* L(\d+)', step_name)
    if match:
        target_row = int(match.group(1))
        sign = match.group(2)
        multiplier_val = float(match.group(3))
        source_row = int(match.group(4))
        
        # Ajuste do multiplicador baseado no sinal
        multiplier = multiplier_val if sign == '+' else -multiplier_val
        
        # Geração das descrições
        desc_long = f'$L_{{{target_row}}} \\leftarrow L_{{{target_row}}} + ({multiplier}) \\cdot L_{{{source_row}}} \\iff L_{{{target_row}}} \\leftarrow L_{{{target_row}}} {sign} {multiplier_val} \\cdot L_{{{source_row}}}$'
        desc_short = f'\\times({multiplier})'
        
        return target_row, source_row, desc_short, desc_long
        
    # Padrão para operações de escalonamento (L_i <- c * L_i)
    match_scale = re.search(r'L(\d+) <- ([\d\.]+) \* L(\d+)', step_name)
    if match_scale:
        target_row = int(match_scale.group(1))
        source_row = int(match_scale.group(3))
        multiplier = float(match_scale.group(2))
        
        if target_row == source_row:
            desc_long = f'$L_{{{target_row}}} \\leftarrow {multiplier} \\cdot L_{{{target_row}}}$'
            desc_short = f'\\times({multiplier})'
            return target_row, None, desc_short, desc_long
            
    # Padrão para operações de permutação (L_i <-> L_j)
    match_swap = re.search(r'Troca de linhas: L(\d+) <-> L(\d+)', step_name)
    if match_swap:
        row1 = int(match_swap.group(1))
        row2 = int(match_swap.group(2))
        desc_long = f'$L_{{{row1}}} \\leftrightarrow L_{{{row2}}}$'
        desc_short = f'L_{{{row1}}} \\leftrightarrow L_{{{row2}}}'
        return row1, row2, desc_short, desc_long

    return None, None, None, None


def display_step_with_annotations(current_matriz, prev_matriz, step_info):
    """
    Exibe a transição entre matrizes com anotações matemáticas detalhadas.
    
    Esta função cria uma visualização rica da transformação matricial,
    mostrando a matriz anterior, a operação aplicada e a matriz resultante
    com formatação LaTeX adequada.
    
    Args:
        current_matriz (np.array): Matriz após a operação
        prev_matriz (np.array): Matriz antes da operação  
        step_info (dict): Dicionário com informações do passo
    """
    step_name = step_info['step_name']
    
    # Tratamento especial para o passo inicial
    if "Início da Eliminação Gaussiana" in step_name:
        st.markdown("**Matriz atual:**")
        latex_matriz = formatar_matriz_latex(current_matriz)
        st.markdown(latex_matriz, unsafe_allow_html=True)
        return

    # Análise da operação
    target_row, source_row, desc_short, desc_long = parse_step_operation(step_name)

    if desc_short:
        # Formatação das matrizes
        latex_prev_str = formatar_matriz_latex(prev_matriz).replace('$$', '')
        latex_current_str = formatar_matriz_latex(current_matriz).replace('$$', '')
        
        if source_row:
            # Exibição da transição com operação de combinação linear
            st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: center; gap: 10px;">
                <div style="font-size: 2em;">
                    {latex_prev_str}
                </div>
                <div style="text-align: center;">
                    <span style="font-size: 2em;">
                        &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;
                    </span>
                    <br>
                    <span style="font-size: 1.5em;">
                        $L_{{{target_row}}} \\leftarrow L_{{{target_row}}} + ({desc_short}) \\cdot L_{{{source_row}}}$
                    </span>
                </div>
                <div style="font-size: 2em;">
                    {latex_current_str}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"**Operação:** {desc_long}")
            
        else:
            # Exibição para operações de troca ou escalonamento
            st.markdown(f"**Matriz anterior:**")
            st.markdown(formatar_matriz_latex(prev_matriz), unsafe_allow_html=True)
            st.markdown(f"**Operação:** {desc_long}")
            st.markdown(f"**Matriz atual:**")
            st.markdown(formatar_matriz_latex(current_matriz), unsafe_allow_html=True)
    else:
        # Fallback para operações não reconhecidas
        st.markdown(f"**Matriz atual:**")
        st.markdown(formatar_matriz_latex(current_matriz), unsafe_allow_html=True)


def formatar_equacao_latex(a, b, c, d):
    """
    Converte coeficientes numéricos em uma equação linear formatada em LaTeX.
    
    Gera representação textual de uma equação da forma ax + by + cz = d,
    tratando casos especiais como coeficientes zero, unitários e negativos.
    
    Args:
        a, b, c (float): Coeficientes das variáveis x, y, z
        d (float): Termo independente
        
    Returns:
        str: Equação formatada em LaTeX
        
    Exemplo:
        >>> formatar_equacao_latex(1, -2, 0, 5)
        'x - 2.0y = 5.0'
    """
    termos = []
    
    # Processamento do coeficiente de x
    if not np.isclose(a, 0):
        if np.isclose(abs(a), 1):
            termos.append(f"{'-' if a < 0 else ''}x")
        else:
            termos.append(f"{a:.1f}x")
    
    # Processamento do coeficiente de y
    if not np.isclose(b, 0):
        sign = ' + ' if b >= 0 and len(termos) > 0 else ' - ' if b < 0 and len(termos) > 0 else '-' if b < 0 else ''
        if np.isclose(abs(b), 1):
            termos.append(f"{sign}y")
        else:
            termos.append(f"{sign}{abs(b):.1f}y")
    
    # Processamento do coeficiente de z
    if not np.isclose(c, 0):
        sign = ' + ' if c >= 0 and len(termos) > 0 else ' - ' if c < 0 and len(termos) > 0 else '-' if c < 0 else ''
        if np.isclose(abs(c), 1):
            termos.append(f"{sign}z")
        else:
            termos.append(f"{sign}{abs(c):.1f}z")
    
    # Montagem da equação final
    equacao_str = "".join(termos).strip()
    if equacao_str.startswith('+'):
        equacao_str = equacao_str[1:].strip()
        
    if not equacao_str:
        return f"0 = {d:.1f}"
        
    return f"{equacao_str} = {d:.1f}"

# =============================================================================
# INTERFACE LATERAL - ENTRADA DE DADOS
# =============================================================================

st.sidebar.header('📥 Entrada de Equações')
st.sidebar.markdown("""
Insira os coeficientes para o sistema de equações lineares da forma:
- **ax + by + cz = d**

Cada equação representa um plano no espaço 3D.
""")

# Equação 1
st.sidebar.markdown('**Equação 1:**')
cols = st.sidebar.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
a1 = cols[0].number_input('', key='a1', format='%.1f', value=1.0, label_visibility='collapsed')
cols[1].markdown('x +')
b1 = cols[2].number_input('', key='b1', format='%.1f', value=1.0, label_visibility='collapsed')
cols[3].markdown('y +')
c1 = cols[4].number_input('', key='c1', format='%.1f', value=1.0, label_visibility='collapsed')
cols[5].markdown('z =')
d1 = cols[6].number_input('', key='d1', format='%.1f', value=6.0, label_visibility='collapsed')

# Equação 2
st.sidebar.markdown('**Equação 2:**')
cols = st.sidebar.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
a2 = cols[0].number_input('', key='a2', format='%.1f', value=1.0, label_visibility='collapsed')
cols[1].markdown('x +')
b2 = cols[2].number_input('', key='b2', format='%.1f', value=-1.0, label_visibility='collapsed')
cols[3].markdown('y +')
c2 = cols[4].number_input('', key='c2', format='%.1f', value=2.0, label_visibility='collapsed')
cols[5].markdown('z =')
d2 = cols[6].number_input('', key='d2', format='%.1f', value=5.0, label_visibility='collapsed')

# Equação 3
st.sidebar.markdown('**Equação 3:**')
cols = st.sidebar.columns([1, 0.3, 1, 0.3, 1, 0.3, 1])
a3 = cols[0].number_input('', key='a3', format='%.1f', value=1.0, label_visibility='collapsed')
cols[1].markdown('x +')
b3 = cols[2].number_input('', key='b3', format='%.1f', value=-1.0, label_visibility='collapsed')
cols[3].markdown('y +')
c3 = cols[4].number_input('', key='c3', format='%.1f', value=-3.0, label_visibility='collapsed')  # Corrigido: era 'c4'
cols[5].markdown('z =')
d3 = cols[6].number_input('', key='d3', format='%.1f', value=-10.0, label_visibility='collapsed')

# =============================================================================
# PROCESSAMENTO E RESOLUÇÃO DO SISTEMA
# =============================================================================

st.sidebar.markdown("---")
if st.sidebar.button("🚀 Resolver Sistema", type="primary"):
    
    # Montagem da matriz de coeficientes
    matriz_list = [
        [a1, b1, c1, d1],
        [a2, b2, c2, d2],
        [a3, b3, c3, d3]
    ]

    matriz = None
    try:
        # Conversão e validação inicial
        matriz = np.array(matriz_list, dtype=float)
        if matriz.shape != (3, 4):
            st.sidebar.warning("Por favor, insira uma matriz com 3 equações e 4 colunas.")
            matriz = None
    except ValueError:
        st.sidebar.error("Erro: Todos os valores da matriz devem ser numéricos.")
        matriz = None

    if matriz is not None:
        try:
            # Validação avançada e resolução
            matriz_validada = validar_matriz_usuario(matriz)
            
            with st.spinner("Resolvendo sistema..."):
                desc, sol, steps_data, solution_point = resolver_sistema_linear_passo_a_passo(
                    matriz_validada, verbose=False
                )
            
            # Armazenamento dos resultados na sessão
            st.session_state['solucao_desc'] = desc
            st.session_state['solucao_valores'] = sol
            st.session_state['steps_data'] = steps_data
            st.session_state['matriz_inicial'] = matriz_validada
            st.session_state['solution_point'] = solution_point
            st.session_state['sistema_resolvido'] = True
            
        except Exception as e:
            st.sidebar.error(f"Erro ao processar matriz: {str(e)}")
            st.sidebar.info("Verifique se a matriz está no formato correto.")
            st.session_state['sistema_resolvido'] = False
            st.session_state['solution_point'] = None
    else:
        st.sidebar.warning("Por favor, insira os valores para as equações antes de resolver.")
        st.session_state['sistema_resolvido'] = False
        st.session_state['solution_point'] = None

# =============================================================================
# EXIBIÇÃO DOS RESULTADOS NA BARRA LATERAL
# =============================================================================

if 'sistema_resolvido' in st.session_state and st.session_state['sistema_resolvido']:
    st.sidebar.markdown("---")
    st.sidebar.header("📋 Resultado:")

    desc = st.session_state['solucao_desc']
    sol = st.session_state['solucao_valores']

    # Classificação do sistema com ícones apropriados
    if "impossível" in desc.lower():
        st.sidebar.error(f"❌ {desc}")
        st.sidebar.markdown("**Sistema Impossível (SI):** Não existe solução que satisfaça todas as equações simultaneamente.")
    elif "indeterminado" in desc.lower():
        st.sidebar.warning(f"⚠️ {desc}")
        st.sidebar.markdown("**Sistema Indeterminado (SPI):** Existem infinitas soluções possíveis.")
    else:
        st.sidebar.success(f"✅ {desc}")
        st.sidebar.markdown("**Sistema Determinado (SPD):** Existe uma única solução.")

    # Exibição das soluções (se existirem)
    if len(sol) > 0:
        st.sidebar.subheader("🎯 Solução:")

        # Mapeamento de variáveis para notação mais amigável
        mapa_variaveis = {
            'x_1': 'x',
            'x_2': 'y',
            'x_3': 'z',
        }

        for i, s in enumerate(sol):
            sol_str = str(s)

            # --- INÍCIO DA ALTERAÇÃO ---
            # Remove a palavra "(livre)" e espaços extras
            sol_str = sol_str.replace('(livre)', '').strip()
            # --- FIM DA ALTERAÇÃO ---

            # Determinação da variável dependente
            if i == 0:
                var_dependente = 'x'
            elif i == 1:
                var_dependente = 'y'
            elif i == 2:
                var_dependente = 'z'
            else:
                var_dependente = f'x_{i+1}'

            # Formatação da solução
            sol_formatada = sol_str
            for var_antiga, var_nova in mapa_variaveis.items():
                sol_formatada = sol_formatada.replace(var_antiga, var_nova)

            if not sol_formatada.startswith(var_dependente + ' ='):
                sol_formatada = f"{var_dependente} = {sol_formatada}"

            st.sidebar.write(f"**{sol_formatada.strip()}**")

# =============================================================================
# INTERFACE PRINCIPAL - VISUALIZAÇÃO E RESULTADOS
# =============================================================================

col1, col2 = st.columns([1, 1])

# -------------------------
# COLUNA 1: SISTEMA DE EQUAÇÕES
# -------------------------

with col1:
    st.header("📝 Sistema de Equações")
    st.markdown("""
    As equações inseridas são exibidas abaixo.
    Cada equação representa um plano no espaço tridimensional.
    """)
    
    # Montagem da matriz para exibição
    matriz_list_display = [
        [a1, b1, c1, d1],
        [a2, b2, c2, d2], 
        [a3, b3, c3, d3]
    ]
    matriz_display = np.array(matriz_list_display, dtype=float)

    # Formatação das equações individuais
    eq1_str = formatar_equacao_latex(a1, b1, c1, d1)
    eq2_str = formatar_equacao_latex(a2, b2, c2, d2)
    eq3_str = formatar_equacao_latex(a3, b3, c3, d3)
    
    # Sistema de equações em LaTeX
    latex_equacoes = f"""
    $$
    \\begin{{cases}}
    {eq1_str} \\\\
    {eq2_str} \\\\
    {eq3_str}
    \\end{{cases}}
    $$
    """
    
    st.markdown(latex_equacoes, unsafe_allow_html=True)

# -------------------------
# COLUNA 2: VISUALIZAÇÃO E ANÁLISE
# -------------------------

with col2:
    st.header("📊 Visualização e Resultados")
    
    if 'sistema_resolvido' in st.session_state and st.session_state['sistema_resolvido']:
        # Recuperação dos dados da sessão
        matriz_inicial = st.session_state['matriz_inicial']
        steps_data = st.session_state['steps_data']
        solution_point = st.session_state['solution_point']
        desc = st.session_state['solucao_desc']

        # Exibição da matriz aumentada inicial
        st.subheader("Matriz Aumentada do Sistema:")
        st.markdown("""
        A matriz aumentada combina os coeficientes das variáveis com os termos independentes,
        separados por uma linha vertical.
        """)
        latex_matriz = formatar_matriz_latex(matriz_inicial)
        st.markdown(latex_matriz, unsafe_allow_html=True)
        
        # Visualização passo a passo
        st.subheader("🌐 Visualização 3D dos Planos - Passo a Passo:")
        st.markdown("""
        Cada aba mostra um passo da eliminação gaussiana e como os planos se transformam geometricamente.
        """)
        
        if len(steps_data) > 0:
            # Criação das abas para cada passo
            tab_names = [f"Passo {i+1}" for i in range(len(steps_data))]
            tabs = st.tabs(tab_names)
            
            prev_matriz = None
            for i, (tab, step_info) in enumerate(zip(tabs, steps_data)):
                with tab:
                    # Obtenção da matriz do passo atual
                    matriz_passo = step_info.get('matriz_float', step_info.get('matriz'))
                    
                    st.markdown(f"**{step_info['step_name']}**")

                    # Exibição da transformação matricial
                    if i > 0:
                        display_step_with_annotations(matriz_passo, prev_matriz, step_info)
                    else:
                        st.markdown("**Matriz inicial:**")
                        latex_passo = formatar_matriz_latex(matriz_passo)
                        st.markdown(latex_passo, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Visualização 3D
                    st.markdown("**Visualização 3D dos Planos:**")
                    try:
                        plot_sol_point = None
                        if "determinado" in desc.lower() and i == len(steps_data) - 1:
                            plot_sol_point = solution_point
                            
                        fig = plotar_planos_3d(
                            matriz_passo, 
                            step=step_info['step_name'], 
                            solution_point=plot_sol_point
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Não foi possível gerar a visualização 3D para este passo.")
                            
                    except Exception as e:
                        st.error(f"Erro na visualização: {str(e)}")
                        st.info("Tentando visualizar apenas as equações válidas...")
                        
                        try:
                            # Filtro de equações válidas (não degeneradas)
                            matriz_filtrada = [
                                row for row in matriz_passo 
                                if not np.allclose(row[:3], 0, atol=1e-10)
                            ]
                            
                            if len(matriz_filtrada) > 0:
                                fig = plotar_planos_3d(
                                    np.array(matriz_filtrada), 
                                    step=step_info['step_name'], 
                                    solution_point=plot_sol_point
                                )
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.warning("Não foi possível gerar visualização para este passo.")

                    # Atualização para o próximo ciclo
                    prev_matriz = matriz_passo
            
        # Seção expandível com processo detalhado        
        with st.expander("🔍 Ver Processo de Resolução Detalhado"):
            st.subheader("Análise Completa da Eliminação Gaussiana:")
            st.markdown("""
            Esta seção mostra todos os passos da eliminação gaussiana de forma sequencial,
            incluindo as operações matriciais.
            """)
            
            prev_matriz_expand = None
            for i, step_info in enumerate(steps_data):
                matriz_passo_expand = step_info.get('matriz_float', step_info.get('matriz'))
                
                st.markdown(f"**Passo {i+1}: {step_info['step_name']}**")
                
                if i > 0:
                    display_step_with_annotations(matriz_passo_expand, prev_matriz_expand, step_info)
                else:
                    st.markdown(formatar_matriz_latex(matriz_passo_expand), unsafe_allow_html=True)
                    
                st.markdown("---")
                prev_matriz_expand = matriz_passo_expand
                
    else:
        st.info("""
        **Como usar:**
        1. Insira os coeficientes das equações na barra lateral
        2. Clique em 'Resolver Sistema' 
        3. Visualize o processo passo a passo e os resultados aqui
        
        A aplicação mostrará tanto a resolução algébrica quanto a interpretação geométrica 3D.
        """)

# =============================================================================
# SEÇÃO DE AJUDA E DOCUMENTAÇÃO
# =============================================================================

st.sidebar.markdown("---")
st.sidebar.header("📚 Ajuda")

with st.sidebar.expander("Como usar a aplicação"):
    st.markdown("""
    ### 🔧 **Entrada de Dados**
    - Insira os coeficientes para cada equação no formato **ax + by + cz = d**
    - Use números decimais (ex: 1.5, -2.7)
    - Deixe coeficientes como zero para variáveis que não aparecem na equação
    
    ### 📊 **Interpretação dos Resultados**
    - **SPD (Sistema Possível Determinado)**: Uma única solução - os três planos se intersectam em um ponto
    - **SPI (Sistema Possível Indeterminado)**: Infinitas soluções - os planos se intersectam em uma linha ou são coincidentes
    - **SI (Sistema Impossível)**: Nenhuma solução - os planos são paralelos ou não têm interseção comum
    
    ### 🌐 **Visualização 3D**
    - Cada aba mostra um passo da eliminação gaussiana
    - Os planos 3D são atualizados conforme a matriz é transformada
    - Observe como as operações simplificam o sistema geometricamente
    - No resultado final (SPD), o ponto de interseção é destacado em dourado
    """)

with st.sidebar.expander("Exemplos de sistemas"):
    st.markdown("""
    ### 🎯 **Sistema com Solução Única (SPD)**
    ```
    x + y + z = 6
    x - y + 2z = 5  
    x - y - 3z = -10
    ```
    
    ### ♾️ **Sistema com Infinitas Soluções (SPI)**
    ```
    x + y - 2z = 6
    2x + 3y + z = 8
    0x + 0y + 0z = 0
    ```
    
    ### ❌ **Sistema Impossível (SI)**
    ```  
    x + y + z = 1
    x + y + z = 2
    2x + 2y + 2z = 5
    ```
    """)

with st.sidebar.expander("Sobre a matemática"):
    st.markdown("""
    ### 📐 **Interpretação Geométrica**
    - Cada equação linear representa um **plano** no espaço 3D
    - A solução do sistema é a **interseção** desses planos
    - Três planos podem se intersectar de diferentes formas:
      - **Um ponto** (solução única)
      - **Uma linha** (infinitas soluções)
      - **Conjunto vazio** (sem solução)
    
    ### 🔢 **Método de Eliminação Gaussiana**
    1. **Escalonamento**: Transformar em matriz triangular superior
    2. **Pivotamento**: Escolher elementos não-nulos como pivôs
    3. **Operações elementares**:
       - Trocar linhas
       - Multiplicar linha por constante não-nula
       - Somar múltiplo de uma linha a outra
    """)

# =============================================================================
# RODAPÉ E INFORMAÇÕES FINAIS
# =============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px;'>
    <h3>🔢 Resolvedor de Sistemas Lineares 3D</h3>
    <p><strong>Desenvolvido com Streamlit </strong></p>
    <p>✨ <em>Ferramenta educacional para a compreensão visual de sistemas lineares</em> ✨</p>
</div>
""", unsafe_allow_html=True)
