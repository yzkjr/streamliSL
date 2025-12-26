"""
=============================================================================
SISTEMA LINEAR LAB 
=============================================================================
Responsável por conectar a interface (UI), a lógica (Backend) e os gráficos.
"""

import streamlit as st

# Módulos do Sistema
from modules.ui import configurar_pagina, aplicar_estilos_css, renderizar_sidebar
from modules.logic import validar_matriz_usuario, resolver_sistema_linear_passo_a_passo, gerar_relatorio_html
from modules.graphics import plotar_planos_3d, plotar_retas_2d
from modules.interface import stream_intro, stream_passo_texto, stream_conclusao

def main():
    # 1. Configuração Inicial
    configurar_pagina()
    aplicar_estilos_css()

    # 2. Renderização da Barra Lateral (Inputs)
    dimensao, matriz_usuario, resolver_btn, var_names = renderizar_sidebar()

    # 3. Área Principal: Cabeçalho e Exibição do Sistema
    st.title(f"Sistema Linear ({dimensao}x{dimensao})")
    st.markdown("### Sistema Definido:")
    
    # Formata o sistema para exibição inicial em LaTeX
    latex_lines = []
    for row in matriz_usuario:
        terms = []
        for j in range(dimensao):
            coef = row[j]
            var = var_names[j] if j < len(var_names) else f"x_{j+1}"
            sinal = "+" if (j > 0 and coef >= 0) else ""
            term = f"{sinal} {coef}{var}"
            terms.append(term)
        latex_lines.append(f"{' '.join(terms)} = {row[-1]}")

    st.latex(r"\begin{cases}" + " \\\\ ".join(latex_lines) + r"\end{cases}")
    st.markdown("---")

    # 4. Execução da Resolução (Ao clicar no botão)
    if resolver_btn:
        executar_resolucao(dimensao, matriz_usuario)

def executar_resolucao(dimensao, matriz_usuario):
    try:
        container = st.container()
        
        # Validação
        is_valido, mensagem = validar_matriz_usuario(matriz_usuario)
        if not is_valido:
            st.error(f"❌ Erro na validação: {mensagem}")
            return # Interrompe a execução
        
        # Cálculo
        desc, sol_textual, steps_data, solution_point = resolver_sistema_linear_passo_a_passo(matriz_usuario, verbose=False)
        
        with container:
            # Intro
            st.write_stream(stream_intro())
            
            # Loop de Passos
            for i, step in enumerate(steps_data):
                st.write_stream(stream_passo_texto(i, step))
                
                # Visualização Gráfica
                if dimensao <= 3:
                    matriz_passo = step.get('matrix') or step.get('matriz') or step.get('matriz_float')
                    mostrar_ponto = solution_point if (i == len(steps_data) - 1) else None

                    fig = None
                    if dimensao == 3:
                        fig = plotar_planos_3d(matriz_passo, step=f"Passo {i+1}", solution_point=mostrar_ponto)
                    else: 
                        fig = plotar_retas_2d(matriz_passo, step=f"Passo {i+1}", solution_point=mostrar_ponto)
                    
                    if fig:
                        st.plotly_chart(fig, use_container_width=True, key=f"plot_{i}")
                
                elif i == 0:
                     st.info(f"ℹ️ Sistemas de ordem {dimensao} não possuem visualização geométrica direta (Hiperplanos em {dimensao}D). Acompanhe a resolução algébrica abaixo.")
                
                st.markdown("---")

            # Conclusão
            st.write_stream(stream_conclusao(desc, sol_textual))
            
            # Exportação
            exibir_exportacao(steps_data, desc, sol_textual)

    except Exception as e:
        st.error(f"❌ Ocorreu um erro inesperado: {str(e)}")

def exibir_exportacao(steps_data, desc, sol_textual):
    st.markdown("### 📥 Exportar Relatório")
    
    col_exp1, _ = st.columns([3, 1])
    with col_exp1:
        st.info(" Baixe o arquivo HTML abaixo, abra no navegador e pressione **Ctrl+P** (Salvar como PDF) para obter o documento.")
    
    html_relatorio = gerar_relatorio_html(steps_data, desc, sol_textual)
    
    st.download_button(
        label="📄 Download da resolução completa (.html)",
        data=html_relatorio,
        file_name="resolucao_sistema.html",
        mime="text/html",
        type="primary"
    )

if __name__ == "__main__":
    main()
