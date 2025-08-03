import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from linear_solver import (
    validar_matriz_usuario, resolver_sistema_linear_passo_a_passo,
    plotar_planos_3d, PLOT_CONFIG
)
import io
import sys
from contextlib import redirect_stdout
import re
import itertools

# Configuração da página
st.set_page_config(
    page_title="Resolvedor de Sistemas Lineares 3D",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título principal
st.title("🔢 Resolvedor de Sistemas Lineares 3x3")
st.markdown("### Uma aplicação interativa para resolver sistemas de equações lineares com visualização 3D passo a passo")

# Função para formatar a matriz em LaTeX
def formatar_matriz_latex(matriz, highlight_row_index=None):
    """
    Formata a matriz para a sintaxe LaTeX de matriz aumentada.
    
    Args:
        matriz (np.array or list): A matriz a ser formatada.
        highlight_row_index (int): O índice da linha a ser destacada.
        
    Returns:
        str: Uma string formatada em LaTeX.
    """
    if matriz is None or len(matriz) == 0 or len(matriz[0]) == 0:
        return ""
    
    matriz_list = matriz.tolist() if isinstance(matriz, np.ndarray) else matriz
    num_rows = len(matriz_list)
    num_cols = len(matriz_list[0])
    
    alignment = 'c' * (num_cols - 1) + '|c'

    rows_latex = []
    for i, row in enumerate(matriz_list):
        formatted_elements = []
        for elem in row:
            # Usa o valor float para garantir precisão antes de formatar
            val = float(elem)
            if np.isclose(val, int(val)):
                formatted_elements.append(str(int(val)))
            else:
                formatted_elements.append(f'{val:.1f}')
        
        row_str = ' & '.join(formatted_elements)
        rows_latex.append(row_str)
    
    matrix_body = ' \\\\ '.join(rows_latex)
    
    latex_string = f'$$\\left(\\begin{{array}}{{{alignment}}} {matrix_body} \\end{{array}}\\right)$$'
    
    return latex_string

def parse_step_operation(step_name):
    """
    Tenta analisar a string do passo para extrair a operação da linha.
    
    Args:
        step_name (str): A string de descrição do passo (e.g., 'L2 <- L2 - 1.0 * L1').
        
    Returns:
        tuple: (target_row, source_row, multiplier, description) ou (None, None, None, None) se não for compatível.
    """
    # Regex para a operação L_i <- L_i + c * L_j
    match = re.search(r'L(\d+) <- L\d+ ([+-]) ([\d\.]+) \* L(\d+)', step_name)
    if match:
        target_row = int(match.group(1))
        sign = match.group(2)
        multiplier_val = float(match.group(3))
        source_row = int(match.group(4))
        
        # Ajusta o multiplicador com base no sinal
        multiplier = multiplier_val if sign == '+' else -multiplier_val
        
        desc_long = f'$L_{{{target_row}}} \\leftarrow L_{{{target_row}}} + ({multiplier}) \\cdot L_{{{source_row}}} \\iff L_{{{target_row}}} \\leftarrow L_{{{target_row}}} {sign} {multiplier_val} \\cdot L_{{{source_row}}}$'
        desc_short = f'\\times({multiplier})'
        
        return target_row, source_row, desc_short, desc_long
        
    # Regex para a operação L_i <- c * L_i
    match_scale = re.search(r'L(\d+) <- ([\d\.]+) \* L(\d+)', step_name)
    if match_scale:
        target_row = int(match_scale.group(1))
        source_row = int(match_scale.group(3))
        multiplier = float(match_scale.group(2))
        
        if target_row == source_row:
            desc_long = f'$L_{{{target_row}}} \\leftarrow {multiplier} \\cdot L_{{{target_row}}}$'
            desc_short = f'\\times({multiplier})'
            return target_row, None, desc_short, desc_long
            
    # Regex para a operação L_i <-> L_j
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
    Exibe a transição de uma matriz para a próxima com anotações de LaTeX.
    
    Args:
        current_matriz (np.array): A matriz após a operação.
        prev_matriz (np.array): A matriz antes da operação.
        step_info (dict): Informações do passo.
    """
    step_name = step_info['step_name']
    
    # Se for o passo inicial, apenas mostre a matriz
    if "Início da Eliminação Gaussiana" in step_name:
        st.markdown("**Matriz atual:**")
        latex_matriz = formatar_matriz_latex(current_matriz)
        st.markdown(latex_matriz, unsafe_allow_html=True)
        return

    target_row, source_row, desc_short, desc_long = parse_step_operation(step_name)

    if desc_short:
        # Tenta formatar a operação com o arrow
        latex_prev = formatar_matriz_latex(prev_matriz)
        latex_current = formatar_matriz_latex(current_matriz)
        
        if source_row:
            # Adiciona um espaço negativo para aproximar a anotação
            full_latex = f"""
            $$\\left(\\begin{{array}}{{ccc|c}}
            {itertools.chain(prev_matriz.tolist())}
            \\end{{array}}\\right)
            \\xrightarrow{{\\substack{{{desc_short}}}\\\\L_{{{target_row}}} \leftarrow L_{{{target_row}}} + {desc_short} \cdot L_{{{source_row}}}}}
            \\left(\\begin{{array}}{{ccc|c}}
            {itertools.chain(current_matriz.tolist())}
            \\end{{array}}\\right)
            $$
            """
            
            # Formato mais simples e robusto sem `itertools`
            latex_prev_str = formatar_matriz_latex(prev_matriz).replace('$$', '')
            latex_current_str = formatar_matriz_latex(current_matriz).replace('$$', '')
            
            # Usando uma tabela de matriz para melhor alinhamento
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
            # Caso de troca de linhas ou escala
            st.markdown(f"**Matriz anterior:**")
            st.markdown(formatar_matriz_latex(prev_matriz), unsafe_allow_html=True)
            st.markdown(f"**Operação:** {desc_long}")
            st.markdown(f"**Matriz atual:**")
            st.markdown(formatar_matriz_latex(current_matriz), unsafe_allow_html=True)
    else:
        # Se a análise falhar, volte ao formato padrão
        st.markdown(f"**Matriz atual:**")
        st.markdown(formatar_matriz_latex(current_matriz), unsafe_allow_html=True)

# Sidebar: Entrada de Equações
st.sidebar.header('📥 Entrada de Equações')

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
c3 = cols[4].number_input('', key='c4', format='%.1f', value=-3.0, label_visibility='collapsed')
cols[5].markdown('z =')
d3 = cols[6].number_input('', key='d3', format='%.1f', value=-10.0, label_visibility='collapsed')

# Botão para resolver (AGORA MAIS PERTO DOS CAMPOS DE ENTRADA)
st.sidebar.markdown("---") # Separador visual para o botão
if st.sidebar.button("🚀 Resolver Sistema", type="primary"):
    matriz_list = [
        [a1, b1, c1, d1],
        [a2, b2, c2, d2],
        [a3, b3, c3, d3]
    ]

    matriz = None
    try:
        matriz = np.array(matriz_list, dtype=float)
        if matriz.shape != (3, 4):
            st.sidebar.warning("Por favor, insira uma matriz com 3 equações e 4 colunas.")
            matriz = None
    except ValueError:
        st.sidebar.error("Erro: Todos os valores da matriz devem ser numéricos.")
        matriz = None

    if matriz is not None:
        try:
            matriz_validada = validar_matriz_usuario(matriz)
            
            with st.spinner("Resolvendo sistema..."):
                desc, sol, steps_data, solution_point = resolver_sistema_linear_passo_a_passo(matriz_validada, verbose=False)
            
            st.session_state['solucao_desc'] = desc
            st.session_state['solucao_valores'] = sol
            st.session_state['steps_data'] = steps_data
            st.session_state['matriz_inicial'] = matriz_validada
            st.session_state['solution_point'] = solution_point
            st.session_state['sistema_resolvido'] = True
            
        except Exception as e:
            st.sidebar.error(f"Erro ao processar matriz: {str(e)}")
            st.sidebar.info("Verifique se a matriz está no formato correto: [[a1,b1,c1,d1],[a2,b2,c2,d2],...]")
            st.session_state['sistema_resolvido'] = False
            st.session_state['solution_point'] = None
    else:
        st.sidebar.warning("Por favor, insira os valores para as equações antes de resolver.")
        st.session_state['sistema_resolvido'] = False
        st.session_state['solution_point'] = None

# Resultados no sidebar (MOVIDOS PARA DEPOIS DO BOTÃO)
if 'sistema_resolvido' in st.session_state and st.session_state['sistema_resolvido']:
    st.sidebar.markdown("---") # Separador visual para os resultados
    st.sidebar.header("📋 Resultado:")
    desc = st.session_state['solucao_desc']
    sol = st.session_state['solucao_valores']

    if "impossível" in desc.lower():
        st.sidebar.error(f"❌ {desc}")
    elif "indeterminado" in desc.lower():
        st.sidebar.warning(f"⚠️ {desc}")
    else:
        st.sidebar.success(f"✅ {desc}")
    
    if len(sol) > 0:
        st.sidebar.subheader("🎯 Solução:")
        mapa_variaveis = {
            'x_1': 'x',
            'x_2': 'y',
            'x_3': 'z',
        }
        
        for i, s in enumerate(sol):
            sol_str = str(s)
            
            if i == 0:
                var_dependente = 'x'
            elif i == 1:
                var_dependente = 'y'
            elif i == 2:
                var_dependente = 'z'
            else:
                var_dependente = f'x_{i+1}'

            sol_formatada = sol_str
            for var_antiga, var_nova in mapa_variaveis.items():
                sol_formatada = sol_formatada.replace(var_antiga, var_nova)
            
            if not sol_formatada.startswith(var_dependente + ' ='):
                sol_formatada = f"{var_dependente} = {sol_formatada}"
            
            st.sidebar.write(f"**{sol_formatada.strip()}**")

# Interface principal
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📝 Equações do Sistema")
    
    matriz_list_display = [
        [a1, b1, c1, d1],
        [a2, b2, c2, d2],
        [a3, b3, c3, d3]
    ]
    matriz_display = np.array(matriz_list_display, dtype=float)

    def formatar_equacao_latex(a, b, c, d):
        termos = []
        
        if not np.isclose(a, 0):
            if np.isclose(abs(a), 1):
                termos.append(f"{'-' if a < 0 else ''}x")
            else:
                termos.append(f"{a:.1f}x")
        
        if not np.isclose(b, 0):
            sign = ' + ' if b >= 0 and len(termos) > 0 else ' - ' if b < 0 and len(termos) > 0 else '-' if b < 0 else ''
            if np.isclose(abs(b), 1):
                termos.append(f"{sign}y")
            else:
                termos.append(f"{sign}{abs(b):.1f}y")
        
        if not np.isclose(c, 0):
            sign = ' + ' if c >= 0 and len(termos) > 0 else ' - ' if c < 0 and len(termos) > 0 else '-' if c < 0 else ''
            if np.isclose(abs(c), 1):
                termos.append(f"{sign}z")
            else:
                termos.append(f"{sign}{abs(c):.1f}z")
        
        equacao_str = "".join(termos).strip()
        if equacao_str.startswith('+'):
            equacao_str = equacao_str[1:].strip()
            
        if not equacao_str:
            return f"0 = {d:.1f}"
            
        return f"{equacao_str} = {d:.1f}"

    eq1_str = formatar_equacao_latex(a1, b1, c1, d1)
    eq2_str = formatar_equacao_latex(a2, b2, c2, d2)
    eq3_str = formatar_equacao_latex(a3, b3, c3, d3)
    
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

with col2:
    st.header("📊 Visualização e Resultados")
    
    if 'sistema_resolvido' in st.session_state and st.session_state['sistema_resolvido']:
        matriz_inicial = st.session_state['matriz_inicial']
        steps_data = st.session_state['steps_data']
        solution_point = st.session_state['solution_point']
        desc = st.session_state['solucao_desc']

        st.subheader("Matriz Aumentada do Sistema:")
        latex_matriz = formatar_matriz_latex(matriz_inicial)
        st.markdown(latex_matriz, unsafe_allow_html=True)
        
        st.subheader("🌐 Visualização 3D dos Planos - Passo a Passo:")
        
        if len(steps_data) > 0:
            tab_names = [f"Passo {i+1}" for i in range(len(steps_data))]
            tabs = st.tabs(tab_names)
            
            prev_matriz = None
            for i, (tab, step_info) in enumerate(zip(tabs, steps_data)):
                with tab:
                    # Obter a matriz do passo atual
                    matriz_passo = step_info.get('matriz_float', step_info.get('matriz'))
                    
                    st.markdown(f"**{step_info['step_name']}**")

                    if i > 0:
                        # Exibir a transição com anotações para passos não iniciais
                        display_step_with_annotations(matriz_passo, prev_matriz, step_info)
                    else:
                        # Passo inicial, apenas exibe a matriz
                        st.markdown("**Matriz atual:**")
                        latex_passo = formatar_matriz_latex(matriz_passo)
                        st.markdown(latex_passo, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    st.markdown("**Visualização 3D:**")
                    try:
                        plot_sol_point = None
                        if "determinado" in desc.lower() and i == len(steps_data) - 1:
                            plot_sol_point = solution_point
                            
                        fig = plotar_planos_3d(matriz_passo, step=step_info['step_name'], solution_point=plot_sol_point)
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("Não foi possível gerar a visualização 3D para este passo.")
                    except Exception as e:
                        st.error(f"Erro na visualização: {str(e)}")
                        st.info("Tentando visualizar apenas as equações válidas...")
                        try:
                            matriz_filtrada = [row for row in matriz_passo if not np.allclose(row[:3], 0, atol=1e-10)]
                            if len(matriz_filtrada) > 0:
                                fig = plotar_planos_3d(np.array(matriz_filtrada), step=step_info['step_name'], solution_point=plot_sol_point)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.warning("Não foi possível gerar visualização para este passo.")

                    # Atualiza a matriz anterior para o próximo loop
                    prev_matriz = matriz_passo
            
        with st.expander("🔍 Ver processo de resolução detalhado"):
            st.subheader("Passos da Eliminação Gaussiana:")
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
        st.info("Insira os valores das equações e clique em 'Resolver Sistema' para ver a visualização e os resultados.")

# Seção de ajuda
st.sidebar.markdown("---")
st.sidebar.header("📚 Ajuda")

with st.sidebar.expander("Como usar"):
    st.markdown("""
    **1. Escolha o método de entrada:**
    - **Manual**: Insira coeficientes um por um
    - **Matriz**: Digite a matriz completa
    - **Exemplos**: Use casos predefinidos
    
    **2. Formato das equações:**
    - $ax + by + cz = d$
    - Cada linha da matriz: $[a, b, c, d]$
    
    **3. Interpretação dos resultados:**
    - **SPD**: Solução única
    - **SPI**: Infinitas soluções
    - **SI**: Sistema impossível
    
    **4. Visualização passo a passo:**
    - Cada aba mostra um passo da resolução
    - Os planos 3D são atualizados conforme a matriz muda
    - Observe como os planos se transformam durante a **eliminação, buscando "zerar" coeficientes abaixo dos pivôs**, o que geometricamente significa que eles se cruzam de forma a simplificar o sistema.
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
    <p>🔢 Resolvedor de Sistemas Lineares 3D | Desenvolvido com Streamlit</p>
    <p>✨ Agora com visualização passo a passo dos planos 3D!</p>
    </div>
    """,
    unsafe_allow_html=True
)

