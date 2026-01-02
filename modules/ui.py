"""
modules/ui.py
Responsável pelos componentes de interface do usuário (CSS, Sidebar, Configurações de Página).
"""

import streamlit as st

def configurar_pagina():
    """Configura o título da aba, ícone e layout."""
    st.set_page_config(
        page_title="SistemaLinearLab",
        page_icon="📐",
        layout="wide"
    )

def aplicar_estilos_css():
    """Aplica CSS customizado para limpar a interface."""
    st.markdown("""
    <style>
        .stDeployButton {display:none;}
        .block-container {padding-top: 2rem;}
        h1 {text-align: center; margin-bottom: 2rem;}
        .stButton button {width: 100%; border-radius: 8px; font-weight: bold;}
        .js-plotly-plot {margin: 0 auto;}
        
        button[data-testid="stNumberInputStepDown"],
        button[data-testid="stNumberInputStepUp"] {
            display: none !important;
        }
        
        div[data-testid="stNumberInput"] input {
            text-align: center;
            padding-right: 0px !important;
        }
    </style>
    """, unsafe_allow_html=True)

def renderizar_sidebar():
    """
    Renderiza a barra lateral de configuração e retorna a dimensão e a matriz preenchida.
    """
    st.sidebar.header('📥 Configuração')

    dimensao = st.sidebar.selectbox(
        "Tamanho do Sistema:",
        options=[2, 3, 4, 5],
        format_func=lambda x: f"{x} Variáveis ({x}x{x})",
        index=1 
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Insira os coeficientes:**")

    var_names = ['x', 'y', 'z', 'w', 'v']
    matriz_input = []
    
    style_texto = "padding-top: 15px; text-align: center; font-weight: bold; white-space: nowrap; font-size: 14px;"
    
    presets = {}
    if dimensao == 2:
        presets = [
            [1,  1,  1,], 
            [2,  3,  7,]   
        ]
        
    elif dimensao == 3:
        presets = [
            [1,  1,  1,  6], 
            [1, -1,  2,  5], 
            [2,  1, -1,  1]  
        ]
    elif dimensao == 4:
        presets = [
            [1,  1, -1,  0, 10],
            [2, -1,  0,  1,  5],
            [1,  0,  3, -1,  7],
            [0,  2, -1,  1,  4]
        ]
    elif dimensao == 5:
        presets = [
            [10, -2, -1,  0,  0, 12],
            [-2,  8, -2, -1,  0,  0],
            [-1, -2,  9, -2,  0,  5],
            [ 0, -1, -2,  7, -1,  0],
            [ 0,  0,  0, -1,  6, 10]
        ]

    for i in range(dimensao):
        st.sidebar.markdown(f"**Equação {i+1}**")
        
        cols_config = []
        for _ in range(dimensao): cols_config.extend([1.8, 0.6]) 
        cols_config.append(1.8) 
        
        cols = st.sidebar.columns(cols_config)
        linha_atual = []
        
        for j in range(dimensao):
            if presets:
                try: val_padrao = float(presets[i][j])
                except IndexError: val_padrao = 0.0
            else:
                val_padrao = 1.0 if i == j else 0.0
                if i == dimensao-1 and j == dimensao-1: val_padrao = 1.0
            
            with cols[j*2]:
                # Se o valor for inteiro (ex: 1.0), mostra como int. Se for float quebrado, mostra float.
                val = st.number_input(
                    f"a_{i}_{j}", 
                    value=val_padrao, 
                    key=f"cell_{dimensao}_{i}_{j}", 
                    label_visibility="collapsed",
                    format="%.4g" # Essa formatação remove zeros à direita automaticamente!
                )
                linha_atual.append(val)
            
            with cols[j*2 + 1]:
                sinal = "=" if j == dimensao - 1 else "+"
                var_name = var_names[j] if j < len(var_names) else f"x{j+1}"
                st.markdown(f"<div style='{style_texto}'>{var_name} {sinal}</div>", unsafe_allow_html=True)
        
        with cols[-1]:
            if presets:
                try: val_d_padrao = float(presets[i][-1])
                except IndexError: val_d_padrao = 0.0
            else: val_d_padrao = float(i+1 + dimensao)

            val_d = st.number_input(
                f"d_{i}", 
                value=val_d_padrao, 
                key=f"res_{dimensao}_{i}", 
                label_visibility="collapsed",
                format="%.4g" # Remove zeros à direita
            )
            linha_atual.append(val_d)
            
        matriz_input.append(linha_atual)
        st.sidebar.markdown("<div style='margin-bottom: 10px'></div>", unsafe_allow_html=True)
        
    st.sidebar.markdown("---")
    resolver_btn = st.sidebar.button("🚀 Resolver Sistema", type="primary")
    
    return dimensao, matriz_input, resolver_btn, var_names
