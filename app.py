"""
=============================================================================
RESOLVEDOR DE SISTEMAS LINEARES
=============================================================================
"""

import streamlit as st
from modules.ui import configurar_pagina, aplicar_estilos_css, renderizar_sidebar
from modules.logic import validar_matriz_usuario, resolver_sistema_linear_passo_a_passo, gerar_relatorio_html
from modules.graphics import plotar_planos_3d, plotar_retas_2d
from modules.interface import stream_intro, stream_passo_texto, stream_conclusao

def main():
    configurar_pagina()
    aplicar_estilos_css()

    dimensao, matriz_usuario, resolver_btn, var_names = renderizar_sidebar()

    st.title(f" Sistema Linear ({dimensao}x{dimensao})")
    st.markdown("### Sistema Definido:")
    
    # --- FUNÇÃO AUXILIAR DE FORMATAÇÃO ---
    def fmt_num(n):
        """Remove .0 se for inteiro"""
        if float(n).is_integer():
            return str(int(n))
        return str(n)

    latex_lines = []
    
    for row in matriz_usuario:
        terms = []
        for j in range(dimensao):
            coef = row[j]
            var = var_names[j] if j < len(var_names) else f"x_{j+1}"
            
            # 1. Remove o .0 (ex: 2.0 -> 2)
            abs_val = abs(coef)
            val_str = fmt_num(abs_val)
            
            # 2. Lógica para ocultar o "1" (ex: 1x -> x)
            # Se o valor absoluto for 1, a string de exibição fica vazia
            if abs_val == 1.0:
                val_display = "" 
            else:
                val_display = val_str 

            # 3. Montagem com sinais (+/-)
            if j == 0:
                # Primeiro termo da equação (sem espaço extra)
                if coef < 0:
                    term = f"-{val_display}{var}" # Ex: -x
                else:
                    term = f"{val_display}{var}"  # Ex: x
            else:
                # Termos seguintes (com espaçamento para o LaTeX ficar bonito)
                if coef >= 0:
                    term = f"+ {val_display}{var}" # Ex: + y
                else:
                    term = f"- {val_display}{var}" # Ex: - z
            
            terms.append(term)
        
        # Formata o resultado (lado direito do igual)
        res_str = fmt_num(row[-1])
        latex_lines.append(f"{' '.join(terms)} = {res_str}")

    st.latex(r"\begin{cases}" + " \\\\ ".join(latex_lines) + r"\end{cases}")
    st.markdown("---")

    if resolver_btn:
        executar_resolucao(dimensao, matriz_usuario)

def executar_resolucao(dimensao, matriz_usuario):
    try:
        container = st.container()
        is_valido, mensagem = validar_matriz_usuario(matriz_usuario)
        
        if not is_valido:
            st.error(f"❌ Erro na validação: {mensagem}")
            return
        
        desc, sol_textual, steps_data, solution_point = resolver_sistema_linear_passo_a_passo(matriz_usuario, verbose=False)
        
        with container:
            st.write_stream(stream_intro())
            
            for i, step in enumerate(steps_data):
                st.write_stream(stream_passo_texto(i, step))
                
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

            st.write_stream(stream_conclusao(desc, sol_textual))
            exibir_exportacao(steps_data, desc, sol_textual)

    except Exception as e:
        st.error(f"❌ Ocorreu um erro inesperado: {str(e)}")

def exibir_exportacao(steps_data, desc, sol_textual):
    st.markdown("### 📥 Exportar Resolução")
    col_exp1, _ = st.columns([3, 1])
    with col_exp1:
        st.info(" Baixe o arquivo HTML abaixo, para obter a resolução.")
    
    html_relatorio = gerar_relatorio_html(steps_data, desc, sol_textual)
    
    st.download_button(
        label="📄 Download da resolução (.html)",
        data=html_relatorio,
        file_name="resolucao_sistema.html",
        mime="text/html",
        type="primary"
    )

if __name__ == "__main__":
    main()
