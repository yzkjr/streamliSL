"""
modules/interface.py
Responsável pelas animações de texto e formatação visual (LaTeX) dos passos.
"""

import streamlit as st
import numpy as np
import time
import re
from fractions import Fraction

# =============================================================================
# PARSING DE TEXTO PARA LATEX
# =============================================================================

def parse_step_operation(step_name):
    """
    Analisa o texto explicativo (ex: 'L2 <- L2 - 2L1') para determinar
    índices de destaque e gerar a seta LaTeX.
    """
    clean_step = step_name
    if "Subtrair" in step_name or "Dividir" in step_name or "Trocar" in step_name:
        # Texto longo já é tratado no logic, aqui é apenas fallback de formatação
        return "", None, None

    clean_step = step_name.replace("Eliminação:", "").strip()
    match = re.search(r'L(\d+)\s*([+-])\s*(?:[\(]?(-?[\d\.]+(?:/[\d\.]+)?)\)?\s*)?L(\d+)', clean_step)
    
    if match:
        target_num = int(match.group(1))
        sign = match.group(2)
        mult_raw = match.group(3)
        source_num = int(match.group(4))
        target_idx = target_num - 1
        source_idx = source_num - 1
        
        mult_val = 1.0
        if mult_raw:
            try:
                if "/" in mult_raw:
                    n, d = mult_raw.split("/")
                    mult_val = float(n) / float(d)
                else: mult_val = float(mult_raw)
            except: mult_val = 1.0
        
        if mult_val < 0:
            sign = '+' if sign == '-' else '-'
            mult_val = abs(mult_val)
        
        if abs(mult_val - round(mult_val)) < 1e-10: 
            val_str = str(int(round(mult_val)))
        else:
            frac = Fraction(mult_val).limit_denominator(10)
            val_str = f"\\dfrac{{{frac.numerator}}}{{{frac.denominator}}}"
            
        mult_display = val_str if val_str != '1' else ''
        latex_note = f"\\to L_{{{target_num}}} {sign} {mult_display} L_{{{source_num}}}"
        return latex_note, target_idx, source_idx
        
    if "<->" in clean_step:
        nums = re.findall(r'L(\d+)', clean_step)
        if len(nums) == 2:
            return f"\\leftrightarrow L_{{{nums[1]}}}", int(nums[0])-1, int(nums[1])-1
            
    return "", None, None

# =============================================================================
# FORMATAÇÃO VISUAL (MATRIZ + ANOTAÇÕES)
# =============================================================================

def formatar_sistema_visual(matriz, highlight_row=None, pivot_row=None, side_note=""):
    """
    Gera o LaTeX para exibir a matriz com anotações laterais (operações de linha)
    e destaque em cores.
    """
    if matriz is None or len(matriz) == 0: return ""
    matriz_list = matriz.tolist() if isinstance(matriz, np.ndarray) else matriz
    if not matriz_list: return ""
    num_rows = len(matriz_list)
    num_cols = len(matriz_list[0])
    
    rows_tex = []
    for i, row in enumerate(matriz_list):
        cells = []
        is_target = (i == highlight_row)
        for j, elem in enumerate(row):
            val_str = ""
            try:
                val = float(elem)
                if abs(val - round(val)) < 1e-10: val_str = f"{int(round(val))}"
                else:
                    frac = Fraction(val).limit_denominator(100)
                    if frac.denominator == 1: val_str = f"{frac.numerator}"
                    else: val_str = f"\\dfrac{{{frac.numerator}}}{{{frac.denominator}}}"
            except: val_str = str(elem)
            
            if pivot_row is not None and i == pivot_row and j == pivot_row:
                 val_str = f"\\boxed{{{val_str}}}"
            if is_target: val_str = f"\\color{{cyan}}{{{val_str}}}"
            cells.append(val_str)
        rows_tex.append(" & ".join(cells))
    
    matrix_body = " \\\\[0.5em] ".join(rows_tex)
    matrix_align = 'c' * (num_cols - 1) + '|c' if num_cols > 1 else 'c'
    latex_matrix = f"\\left[\\begin{{array}}{{{matrix_align}}} {matrix_body} \\end{{array}}\\right]"

    notes_tex = []
    for i in range(num_rows):
        if i == highlight_row and side_note: notes_tex.append(f"\\color{{cyan}}{{{side_note}}}") 
        else: notes_tex.append("\\phantom{\\to L_1}") 
    notes_body = " \\\\[0.5em] ".join(notes_tex)
    latex_notes = f"\\begin{{array}}{{l}} {notes_body} \\end{{array}}"
    return f"$$ {latex_matrix} \\quad {latex_notes} $$"

# =============================================================================
# ANIMAÇÕES DE STREAMING
# =============================================================================

def typewriter_effect(text, delay=0.01): 
    for char in text:
        yield char
        time.sleep(delay)

def stream_intro():
    yield from typewriter_effect("### 🤖 Iniciando Resolução...\n\n")
    time.sleep(0.3)
    yield from typewriter_effect("Método: **Eliminação de Gauss-Jordan**\n")

def stream_passo_texto(i, step_info):
    texto_explicativo = step_info.get('action', step_info.get('step_name', 'Passo'))
    matriz_float = step_info.get('matriz_float', step_info.get('matriz'))
    matriz_latex = step_info.get('matriz_latex')
    acao_latex = step_info.get('acao_latex')

    side_note, target_idx, source_idx = parse_step_operation(texto_explicativo)

    yield from typewriter_effect(f"\n---\n#### 🔹 Passo {i+1}\n")
    yield from typewriter_effect(f"{texto_explicativo}\n\n")

    if acao_latex:
        yield f"$$ {acao_latex} $$\n\n"

    if matriz_latex:
        yield f"$$ {matriz_latex} $$\n"
    else:
        # Fallback caso o latex não venha pronto do backend
        latex_final = formatar_sistema_visual(
            matriz_float,
            highlight_row=target_idx,
            pivot_row=source_idx,
            side_note=side_note if side_note else ""
        )
        yield latex_final + "\n"

    time.sleep(0.5)

def stream_conclusao(desc_final, solucao_textual):
    yield "\n---\n### ✅ Resultado Final\n"
    
    if "impossível" in desc_final.lower():
        msg = "Sistema Impossível (Não há solução)."
    elif "indeterminado" in desc_final.lower():
        msg = "Sistema Possível e Indeterminado (Infinitas soluções)."
    else:
        msg = "Sistema Possível e Determinado (Solução Única)."
        
    yield from typewriter_effect(f"**Classificação:** {msg}\n\n")
    
    if len(solucao_textual) > 0:
        yield from typewriter_effect("**Solução Encontrada:**\n")
        latex_sol = "$$\\begin{cases} "
        lines = []
        for s in solucao_textual:
            texto = s
            # Limpeza básica de segurança
            if not any(c.isalpha() for c in s):
                try:
                    val = eval(s) if "/" in s else float(s)
                    texto = f"{val:.2f}"
                except: pass
            clean_s = texto.replace("x_1", "x").replace("x_2", "y").replace("x_3", "z").replace("x_4", "w").replace("x_5", "v")
            lines.append(clean_s)
        latex_sol += " \\\\ ".join(lines)
        latex_sol += " \\end{cases}$$"
        yield latex_sol
