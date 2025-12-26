"""
modules/logic.py
Responsável pela lógica matemática (Gauss-Jordan), validação e geração de relatórios.
"""

from __future__ import annotations
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np

# =============================
# Formatação Numérica
# =============================

def _is_close_int(x: float, tol: float = 1e-12) -> bool:
    return abs(x - round(x)) <= tol

def num_to_text_fmt(x: float) -> str:
    """Formata números para texto simples (txt/log)."""
    if _is_close_int(x):
        return str(int(round(x)))
    return f"{x:.2f}"

def num_to_latex(x: float, *, tol: float = 1e-12, max_den: int = 10000) -> str:
    """Converte float para LaTeX (\dfrac)."""
    if _is_close_int(x, tol):
        return str(int(round(x)))

    frac = Fraction(x).limit_denominator(max_den)
    
    if abs(float(frac) - x) <= 1e-9:
        a, b = frac.numerator, frac.denominator
        if b == 1: return str(a)
        if b < 0: a, b = -a, -b
        return r"\dfrac{%d}{%d}" % (a, b)
    
    return f"{x:.2f}"

def format_coef_latex(c: float) -> str:
    if abs(c - 1.0) < 1e-12: return ""
    if abs(c + 1.0) < 1e-12: return "-"
    return num_to_latex(c)

# =============================
# Geração da Matriz (LaTeX)
# =============================

def augmented_matrix_to_latex(M: List[List[float]], highlight_rows: Set[int] = None, pivot_pos: Tuple[int, int] = None) -> str:
    if not M: return r"\left[\right]"
    rows = len(M)
    cols = len(M[0])
    n = cols - 1
    col_spec = ("c" * n) + "|c" 
    body_lines = []
    
    for r in range(rows):
        cells = []
        row_color_prefix = r"\color{orange} " if (highlight_rows and r in highlight_rows) else ""
        for c in range(cols):
            val_latex = num_to_latex(M[r][c])
            if pivot_pos and r == pivot_pos[0] and c == pivot_pos[1]:
                val_latex = rf"\boxed{{{val_latex}}}"
            cells.append(f"{row_color_prefix}{val_latex}")
            
        left = " & ".join(cells[:n])
        right = cells[n]
        body_lines.append(f"{left} & {right}")
    
    body = r"\\[0.5em] ".join(body_lines)
    return rf"\left[\begin{{array}}{{{col_spec}}}{body}\end{{array}}\right]"

# =============================
# Operações (LaTeX)
# =============================

def _op_elim_latex(i: int, k: int, f: float) -> str:
    sign = "-" if f > 0 else "+"
    val = abs(f)
    coef_str = format_coef_latex(val)
    rhs_part = rf"L_{k+1}" if coef_str == "" else rf"{coef_str}L_{k+1}"
    return rf"L_{i+1} \leftarrow L_{i+1} {sign} {rhs_part}"

def _op_div_latex(k: int, pivot: float) -> str:
    if abs(pivot) < 1e-12: return "" 
    inv_pivot = 1.0 / pivot
    coef = format_coef_latex(inv_pivot)
    return rf"L_{k+1} \leftarrow {coef}L_{k+1}"

def _op_swap_latex(a: int, b: int) -> str:
    return rf"L_{a+1} \leftrightarrow L_{b+1}"


# =============================
# Solver Lógico (Core)
# =============================

@dataclass
class Step:
    step: int
    action: str
    matrix: List[List[float]]
    action_latex: str
    matrix_latex: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "action": self.action,
            "matrix": self.matrix,
            "acao_latex": self.action_latex,
            "matriz_latex": self.matrix_latex,
        }

def eliminar_gaussiana(matriz_aumentada: List[List[float]]) -> Tuple[List[str], List[Dict[str, Any]], str, Optional[List[float]]]:
    matriz_aumentada = [row[:] for row in matriz_aumentada]
    n = len(matriz_aumentada)
    if n == 0: return [], [], "Sistema vazio", None
    
    steps: List[Step] = []
    step_id = 1

    def add_step(action_text: str, action_latex: str, highlight_rows: Set[int] = None, pivot_pos: Tuple[int, int] = None):
        nonlocal step_id
        mat_copy = [r[:] for r in matriz_aumentada]
        mat_latex = augmented_matrix_to_latex(mat_copy, highlight_rows=highlight_rows, pivot_pos=pivot_pos)
        steps.append(Step(step_id, action_text, mat_copy, action_latex, mat_latex))
        step_id += 1

    add_step("Matriz aumentada inicial do sistema.", "")

    # --- FASE 1: Gauss ---
    for k in range(n):
        pivot_row = max(range(k, n), key=lambda r: abs(matriz_aumentada[r][k]))
        if abs(matriz_aumentada[pivot_row][k]) < 1e-12: continue
        
        if pivot_row != k:
            matriz_aumentada[k], matriz_aumentada[pivot_row] = matriz_aumentada[pivot_row], matriz_aumentada[k]
            texto = f"Trocar a linha **{k+1}** com a linha **{pivot_row+1}**."
            add_step(texto, _op_swap_latex(k, pivot_row), highlight_rows={k, pivot_row})

        pivot = matriz_aumentada[k][k]

        if abs(pivot - 1.0) > 1e-12:
            for j in range(k, n + 1):
                matriz_aumentada[k][j] /= pivot
            p_str = num_to_text_fmt(pivot)
            texto = f"Dividir a linha **{k+1}** por **{p_str}** para obter o pivô 1."
            add_step(texto, _op_div_latex(k, pivot), highlight_rows={k}, pivot_pos=(k, k))

        for i in range(k + 1, n):
            f = matriz_aumentada[i][k]
            if abs(f) < 1e-12: continue
            for j in range(k, n + 1):
                matriz_aumentada[i][j] -= f * matriz_aumentada[k][j]
            f_str = num_to_text_fmt(f)
            texto = f"Subtrair da linha **{i+1}** a linha **{k+1}** multiplicada por **{f_str}**."
            add_step(texto, _op_elim_latex(i, k, f), highlight_rows={i}, pivot_pos=(k, k))

    # --- FASE 2: Jordan ---
    for k in range(n - 1, 0, -1):
        if abs(matriz_aumentada[k][k]) < 1e-12: continue 
        for i in range(k - 1, -1, -1):
            f = matriz_aumentada[i][k]
            if abs(f) < 1e-12: continue
            for j in range(k, n + 1):
                matriz_aumentada[i][j] -= f * matriz_aumentada[k][j]
            f_str = num_to_text_fmt(f)
            texto = f"Subtrair da linha **{i+1}** a linha **{k+1}** multiplicada por **{f_str}**."
            add_step(texto, _op_elim_latex(i, k, f), highlight_rows={i}, pivot_pos=(k, k))

    # --- Resultados ---
    for i in range(n):
        if all(abs(matriz_aumentada[i][j]) < 1e-10 for j in range(n)) and abs(matriz_aumentada[i][n]) > 1e-10:
            return [], [s.to_dict() for s in steps], "Sistema Impossível (Sem Solução)", None

    rank_a = np.linalg.matrix_rank(np.array([row[:n] for row in matriz_aumentada], dtype=float))
    
    sol_txt = []
    ponto_plot = None
    
    if rank_a < n:
        classification = "Sistema Possível e Indeterminado (Infinitas Soluções)"
        sol_txt = resolver_spi_parametrico(matriz_aumentada)
    else:
        classification = "Sistema Possível e Determinado (Solução Única)"
        x = [matriz_aumentada[i][n] for i in range(n)]
        ponto_plot = x[:3] if len(x) >= 3 else None
        for i, val in enumerate(x):
            val_fmt = num_to_text_fmt(val)
            sol_txt.append(f"x_{i+1} = {val_fmt}")

    add_step("Matriz na forma escalonada reduzida.", "")
    return sol_txt, [s.to_dict() for s in steps], classification, ponto_plot


# =============================
# Helpers: SPI, Validação e RELATÓRIO
# =============================

def resolver_spi_parametrico(M_final: List[List[float]]) -> List[str]:
    rows = len(M_final)
    cols = len(M_final[0])
    num_vars = cols - 1
    var_names = ['x', 'y', 'z', 'w', 'v'][:num_vars]
    
    pivots = {} 
    pivot_cols = set()
    for r in range(rows):
        for c in range(num_vars):
            if abs(M_final[r][c]) > 1e-10:
                pivots[r] = c
                pivot_cols.add(c)
                break
    
    solution_lines = [""] * num_vars
    for j in range(num_vars):
        if j not in pivot_cols:
            solution_lines[j] = f"{var_names[j]}"

    for r in range(rows - 1, -1, -1):
        if r not in pivots: continue 
        c_pivot = pivots[r]
        pivot_val = M_final[r][c_pivot]
        rhs = M_final[r][-1]
        
        terms = []
        const_val = rhs / pivot_val
        if abs(const_val) > 1e-10:
            terms.append(num_to_text_fmt(const_val))
            
        for c_other in range(c_pivot + 1, num_vars):
            coef = M_final[r][c_other]
            if abs(coef) > 1e-10:
                val = -coef / pivot_val
                abs_val = abs(val)
                sign = "+" if val > 0 else "-"
                coef_str = "" if abs(abs_val - 1.0) < 1e-10 else num_to_text_fmt(abs_val)
                terms.append(f"{sign} {coef_str}{var_names[c_other]}")
        
        if not terms: solution_lines[c_pivot] = "0"
        else:
            res = " ".join(terms).strip()
            if res.startswith("+ "): res = res[2:]
            solution_lines[c_pivot] = res

    final_output = []
    for i in range(num_vars):
        val = solution_lines[i] if solution_lines[i] else "Indefinido"
        final_output.append(f"{var_names[i]} = {val}")
        
    return final_output

def validar_matriz_usuario(matriz: List[List[float]]) -> Tuple[bool, str]:
    if not isinstance(matriz, list) or not matriz: return False, "Matriz inválida."
    n = len(matriz)
    if len(matriz[0]) != n + 1: return False, f"Formato inválido."
    for i, row in enumerate(matriz):
        for j, val in enumerate(row):
            try:
                if not np.isfinite(float(val)): raise ValueError
            except: return False, "Valor inválido."
    if n < 2 or n > 5: return False, "Suporte apenas 2x2 a 5x5."
    return True, "OK"

def resolver_sistema_linear_passo_a_passo(matriz: List[List[float]], verbose: bool = False, **_kwargs: Any) -> Tuple[str, List[str], List[Dict[str, Any]], Optional[List[float]]]:
    sol_txt, steps, classificacao, ponto = eliminar_gaussiana(matriz)
    return classificacao, sol_txt, steps, ponto

# --- GERADOR DE RELATÓRIO HTML ---

def gerar_relatorio_html(steps: List[Dict], classificacao: str, solucao_textual: List[str]) -> str:
    """Gera um relatório HTML completo com suporte a LaTeX (MathJax)."""
    html = """
    <!DOCTYPE html>
    <html lang="pt-br">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório de Resolução - SistemaLinearLab</title>
        <script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
        <script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
        <style>
            body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 40px; max-width: 900px; margin: 0 auto; color: #333; }
            h1 { text-align: center; color: #2c3e50; border-bottom: 2px solid #2c3e50; padding-bottom: 10px; }
            .step-container { background: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 30px; border-left: 5px solid #3498db; box-shadow: 0 2px 5px rgba(0,0,0,0.05); }
            .step-title { font-weight: bold; font-size: 1.2em; color: #2980b9; margin-bottom: 10px; }
            .action-text { font-size: 1.1em; margin-bottom: 15px; }
            .math-block { overflow-x: auto; padding: 10px 0; text-align: center; }
            .result-box { background: #e8f8f5; border: 2px solid #2ecc71; padding: 20px; border-radius: 8px; margin-top: 40px; }
            .footer { text-align: center; margin-top: 50px; font-size: 0.8em; color: #7f8c8d; }
            @media print {
                body { padding: 0; }
                .step-container { break-inside: avoid; }
            }
        </style>
    </head>
    <body>
        <h1>SistemaLinearLab - Relatório de Resolução</h1>
        <p style="text-align: center;">Este documento foi gerado automaticamente. Para salvar como PDF, use a função de imprimir do navegador (Ctrl+P).</p>
        <br>
    """

    for step in steps:
        action_clean = step['action'].replace("**", "")
        html += f"""
        <div class="step-container">
            <div class="step-title">Passo {step['step']}</div>
            <div class="action-text">{action_clean}</div>
        """
        if step.get('acao_latex'):
            html += f"""<div class="math-block">$$ {step['acao_latex']} $$</div>"""
        if step.get('matriz_latex'):
            html += f"""<div class="math-block">$$ {step['matriz_latex']} $$</div>"""
        html += "</div>"

    html += f"""
        <div class="result-box">
            <h2 style="color: #27ae60;">✅ Resultado Final</h2>
            <p><strong>Classificação:</strong> {classificacao}</p>
    """
    
    if solucao_textual:
        latex_sol = r"\begin{cases} " + r" \\ ".join(solucao_textual) + r" \end{cases}"
        html += f"""
            <p><strong>Solução Encontrada:</strong></p>
            <div class="math-block">$$ {latex_sol} $$</div>
        """
        
    html += """
        </div>
        <div class="footer">Gerado por SistemaLinearLab</div>
    </body>
    </html>
    """
    
    return html
