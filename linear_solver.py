import numpy as np
from fractions import Fraction
import ast
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# GLOBAL CONFIGURATIONS
# =============================================================================

PLOT_CONFIG = {
    'mesh_range': (-12, 12),
    'n_points': 80,
    'tolerance': 1e-10,
    'opacity': 0.6,
    'colors': px.colors.qualitative.Plotly,
    'background_color': 'white',
    'fig_size': {'width': 900, 'height': 700},
    'tick_step': 1,
    'axis_line_width': 3,
    'cone_size': 1.0,
    'label_offset': 1.0,
}

TOL = 1e-10

# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validar_coeficientes(*args):
    """Verifica se todos os coeficientes são números válidos (int ou float)."""
    for i, val in enumerate(args):
        if not isinstance(val, (int, float)):
            raise ValueError(f"Coeficiente {i+1} inválido: {val}. Deve ser um número.")
    return True

def validar_matriz_usuario(matriz):
    """
    Verifica se a entrada é lista/tuple/ndarray e converte para numpy.ndarray de floats;
    Garante que seja bidimensional, não vazia e com 4 colunas.
    """
    if not isinstance(matriz, (list, tuple, np.ndarray)):
        raise ValueError("A entrada deve ser uma lista/tuple/array.")
    try:
        arr = np.array(matriz, dtype=float)
    except Exception as e:
        raise ValueError(f"Erro ao processar matriz: {e}")

    if arr.ndim != 2:
        raise ValueError("A matriz deve ser bidimensional.")
    m, n = arr.shape
    if n != 4:
        raise ValueError(f"A matriz deve ter 4 colunas (a, b, c, d). Recebido: {n}")
    if m == 0:
        raise ValueError("A matriz não pode ser vazia.")
    return arr

# =============================================================================
# MATH & HELPER FUNCTIONS
# =============================================================================

def MDC(a, b):
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a

def calcular_mdc_array(linha):
    elems = []
    for x in linha:
        if abs(x) > TOL:
            if np.isclose(x, round(x), atol=TOL):
                elems.append(int(round(x)))
            else:
                return 1
    if not elems:
        return 1
    m = elems[0]
    for e in elems[1:]:
        m = MDC(m, e)
    return abs(m)

def encontrar_primeiro_nao_zero(linha):
    for j, x in enumerate(linha):
        if abs(x) > TOL:
            return x, j
    return None, None

def to_float_array(A):
    return A.astype(float, copy=False)

def dividir_por_mdc(linha):
    m = calcular_mdc_array(linha)
    if m != 1:
        linha = linha / m
        linha = np.where(np.isclose(linha, np.round(linha), atol=TOL), np.round(linha), linha)
    return linha

def frac_str(num, den):
    if abs(den) < TOL:
        return "Indefinido"
    f = Fraction(num).limit_denominator() / Fraction(den).limit_denominator()
    n, d = f.numerator, f.denominator
    if d == 1:
        return str(n)
    if d < 0:
        n, d = -n, -d
    return f"{n}/{d}"

def var_label(index):
    """Retorna o rótulo da variável (x_1, x_2, etc.)"""
    return f"x_{index + 1}"

# =============================================================================
# 3D PLOTTING
# =============================================================================

def _calculate_plane_mesh(equation, meshes, tol):
    a, b, c, d = equation
    xx_xy, yy_xy, yy_yz, zz_yz, xx_xz, zz_xz = meshes
    if abs(c) > tol:
        zz = (d - a*xx_xy - b*yy_xy) / c
        return {'x': xx_xy, 'y': yy_xy, 'z': zz}
    if abs(b) > tol:
        yy = (d - a*xx_xz - c*zz_xz) / b
        return {'x': xx_xz, 'y': yy, 'z': zz_xz}
    if abs(a) > tol:
        xx = (d - b*yy_yz - c*zz_yz) / a
        return {'x': xx, 'y': yy_yz, 'z': zz_yz}
    return None

def plotar_planos_3d(matriz, step=None, config=None, solution_point=None):
    """
    Plota planos 3D a partir de uma matriz de coeficientes.

    Args:
        matriz (np.ndarray): Matriz de coeficientes [a, b, c, d].
        step (str, optional): Descrição do passo atual. Defaults to None.
        config (dict, optional): Configurações de plotagem. Defaults to None.
        solution_point (list, optional): Coordenadas [x, y, z] do ponto de solução. Defaults to None.

    Returns:
        plotly.graph_objects.Figure: A figura Plotly.
    """
    if config is None:
        config = PLOT_CONFIG
    if not isinstance(matriz, np.ndarray):
        matriz = np.array(matriz, dtype=float)
    m, n = matriz.shape
    if n != 4:
        return None

    cfg = config
    vals = np.linspace(cfg['mesh_range'][0], cfg['mesh_range'][1], cfg['n_points'])
    meshes = np.meshgrid(vals, vals)
    all_meshes = (meshes[0], meshes[1], meshes[0], meshes[1], meshes[0], meshes[1])

    fig = go.Figure()
    for i, (a, b, c, d) in enumerate(matriz):
        if np.allclose([a, b, c], 0, atol=cfg['tolerance']):
            continue
        mesh = _calculate_plane_mesh((a, b, c, d), all_meshes, cfg['tolerance'])
        if mesh is None:
            continue
        color = cfg['colors'][i % len(cfg['colors'])]
        fig.add_trace(go.Surface(
            x=mesh['x'], y=mesh['y'], z=mesh['z'],
            opacity=cfg['opacity'],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"Plano {i+1}: {a:.1f}x + {b:.1f}y + {c:.1f}z = {d:.1f}"
        ))

    # Adiciona o ponto de solução se ele for fornecido
    if solution_point is not None and len(solution_point) == 3 and all(isinstance(val, (int, float)) for val in solution_point):
        x_sol, y_sol, z_sol = solution_point

        fig.add_trace(go.Scatter3d(
            x=[x_sol],
            y=[y_sol],
            z=[z_sol],
            mode='markers',
            marker=dict(
                size=10,
                color='gold',
                symbol='circle',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name=f"Solução: ({x_sol:.2f}, {y_sol:.2f}, {z_sol:.2f})",
            hoverinfo='name+x+y+z'
        ))

        fig.update_layout(
            scene=dict(
                annotations=[
                    dict(
                        x=x_sol,
                        y=y_sol,
                        z=z_sol,
                        text=f"Solução: ({x_sol:.1f}, {y_sol:.1f}, {z_sol:.1f})",
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=-20,
                        font=dict(size=12, color="white"),
                        bgcolor="rgba(0,0,0,0.5)"
                    )
                ]
            )
        )

    # CORREÇÃO AQUI: Remove o range, mantendo o dtick e o tickfont
    fig.update_layout(
        title_text=f"Visualização 3D: {step}" if step else "Visualização 3D",
        width=cfg['fig_size']['width'],
        height=cfg['fig_size']['height'],
        scene=dict(
            aspectmode='cube',
            xaxis=dict(
                # dtick=1, # <--- REMOVA ESTA LINHA
                tickfont=dict(size=14),
                nticks=10 # <--- ADICIONE ESTA LINHA: Sugere aproximadamente 10 ticks
            ),
            yaxis=dict(
                # dtick=1, # <--- REMOVA ESTA LINHA
                tickfont=dict(size=14),
                nticks=10 # <--- ADICIONE ESTA LINHA: Sugere aproximadamente 10 ticks
            ),
            zaxis=dict(
                # dtick=1, # <--- REMOVA ESTA LINHA
                tickfont=dict(size=14),
                nticks=10 # <--- ADICIONE ESTA LINHA: Sugere aproximadamente 10 ticks
            )
        )
    )
    return fig
# =============================================================================
# GAUSSIAN & GAUSS-JORDAN ELIMINATION
# =============================================================================

def Eliminar(A, i, j, cnt, steps_data, verbose=True):
    A_copy = A.copy()
    elem = A_copy[i, j]
    p = A_copy[j - cnt, j]

    if abs(elem) < TOL:
        return

    g = abs(MDC(p, elem))

    # Constrói a string da operação de forma otimizada
    operation_str = ""
    if g != 1:
        A_copy[i] = (p/g)*A_copy[i] - (elem/g)*A_copy[j - cnt]
        num_str = frac_str(elem, g)
        den_str = frac_str(p, g)
        if den_str == "1": # Se o denominador simplificado for 1
            operation_str = f"L{i+1} - {num_str}L{j-cnt+1}"
        else:
            operation_str = f"L{i+1} - ({num_str}/{den_str})L{j-cnt+1}"
    else:
        A_copy[i] = p*A_copy[i] - elem*A_copy[j - cnt]
        # Se elem/p for 1.0/1.0, exiba apenas 1.0
        if abs(p) > TOL: # Evita divisão por zero
            fraction_val = elem / p
            if np.isclose(fraction_val, 1.0, atol=TOL):
                operation_str = f"L{i+1} - L{j-cnt+1}"
            elif np.isclose(fraction_val, -1.0, atol=TOL):
                operation_str = f"L{i+1} + L{j-cnt+1}"
            else:
                operation_str = f"L{i+1} - ({elem:.1f}/{p:.1f})L{j-cnt+1}"
        else:
            operation_str = f"L{i+1} - ({elem:.1f}/{p:.1f})L{j-cnt+1}" # Fallback, should not happen often if p is a pivot

    A_copy[i] = dividir_por_mdc(A_copy[i])

    if verbose:
        print_matriz(A_copy)

    steps_data.append({'matriz': A_copy.copy(), 'step_name': f"Eliminação: {operation_str}"})

    A[:] = A_copy[:]

def Reduzir(A, steps_data, verbose=True):
    A_copy = A.copy()
    m, n = A_copy.shape
    cnt = 0

    if verbose:
        print("=== Gauss ===")
        print_matriz(A_copy)

    steps_data.append({'matriz': A_copy.copy(), 'step_name': "Início da Eliminação Gaussiana"})

    for j in range(n):
        if j - cnt >= m:
            break
        if abs(A_copy[j-cnt, j]) < TOL:
            for i in range(j+1-cnt, m):
                if abs(A_copy[i, j]) > TOL:
                    A_copy[[j-cnt, i]] = A_copy[[i, j-cnt]]
                    if verbose:
                        print_matriz(A_copy)
                    steps_data.append({'matriz': A_copy.copy(), 'step_name': f"Permutação: L{j-cnt+1} <-> L{i+1}"})
                    break
            else:
                cnt += 1
                continue
        for i in range(j+1-cnt, m):
            Eliminar(A_copy, i, j, cnt, steps_data, verbose)

    if verbose:
        print("=== Escalonada ===")
        print_matriz(A_copy)

    steps_data.append({'matriz': A_copy.copy(), 'step_name': "Matriz Escalonada"})

    A[:] = A_copy[:]

def remover_linhas_nulas(A, steps_data, verbose=True):
    A_copy = A.copy()
    keep = [row for row in A_copy if not np.allclose(row, 0, atol=TOL)]
    B = np.array(keep, dtype=float)

    if verbose:
        print("=== Sem nulas ===")
        print_matriz(B)

    steps_data.append({'matriz': B.copy(), 'step_name': "Remoção de Linhas Nulas"})

    return B

def verificar_linhas_inconsistentes(A):
    for row in A:
        if np.allclose(row[:-1], 0, atol=TOL) and abs(row[-1]) > TOL:
            return True
    return False

def eliminar_acima_com_pivo(A, steps_data, verbose=True):
    A_copy = A.copy()
    m, _ = A_copy.shape

    if verbose:
        print("=== Regressiva ===")

    for i in range(m-1, -1, -1):
        pv, pc = encontrar_primeiro_nao_zero(A_copy[i])

        if pv is None:
            continue

        A_copy[i] = dividir_por_mdc(A_copy[i])

        p = A_copy[i, pc]

        for j in range(i-1, -1, -1):
            e = A_copy[j, pc]

            if abs(e) < TOL:
                continue

            g = abs(MDC(p, e))

            # Constrói a string da operação de forma otimizada para eliminação regressiva
            operation_str = ""
            if g != 1:
                A_copy[j] = (p/g)*A_copy[j] - (e/g)*A_copy[i]
                num_str = frac_str(e, g)
                den_str = frac_str(p, g)
                if den_str == "1":
                    operation_str = f"L{j+1} - {num_str}L{i+1}"
                else:
                    operation_str = f"L{j+1} - ({num_str}/{den_str})L{i+1}"
            else:
                A_copy[j] = p*A_copy[j] - e*A_copy[i]
                if abs(p) > TOL:
                    fraction_val = e / p
                    if np.isclose(fraction_val, 1.0, atol=TOL):
                        operation_str = f"L{j+1} - L{i+1}"
                    elif np.isclose(fraction_val, -1.0, atol=TOL):
                        operation_str = f"L{j+1} + L{i+1}"
                    else:
                        operation_str = f"L{j+1} - ({e:.1f}/{p:.1f})L{i+1}"
                else:
                    operation_str = f"L{j+1} - ({e:.1f}/{p:.1f})L{i+1}" # Fallback

            A_copy[j] = dividir_por_mdc(A_copy[j])

            if verbose:
                print_matriz(A_copy)

            steps_data.append({'matriz': A_copy.copy(), 'step_name': f"Eliminação Regressiva: {operation_str}"})

    A[:] = A_copy[:]

def Normalizar(A):
    """
    Normaliza a matriz para a forma escalonada reduzida.
    Retorna uma tupla: (matriz_de_strings_para_exibicao, matriz_de_floats_para_plotagem)
    """
    if A.size == 0:
        return np.array([], dtype=object), np.array([], dtype=float)

    A_copy = A.copy()
    m, n = A_copy.shape
    R_str = np.zeros((m, n), dtype=object)
    R_float = np.zeros((m, n), dtype=float)

    for i in range(m):
        pv, pc = encontrar_primeiro_nao_zero(A_copy[i])

        if pv is None:
            R_str[i, :] = "0"
            R_float[i, :] = 0.0
            continue

        L = dividir_por_mdc(A_copy[i])
        pv = L[pc]

        # Matriz de floats para plotagem e cálculos
        L_float = L / pv
        L_float = np.where(np.isclose(L_float, np.round(L_float), atol=TOL), np.round(L_float), L_float)
        R_float[i] = L_float

        # Matriz de strings para exibição da solução e passos
        for j in range(n):
            v = L[j]

            if abs(v) < TOL:
                R_str[i, j] = "0"
            elif j == pc:
                R_str[i, j] = "1"
            else:
                R_str[i, j] = frac_str(v, pv)

    return R_str, R_float

# =============================================================================
# CLASSIFICATION & SOLUTION
# =============================================================================

def classificar_sistema(A):
    m, n = A.shape

    if verificar_linhas_inconsistentes(A):
        return "SI", "Sistema impossível (SI)"

    pivs = sum(not np.allclose(A[i, :-1], 0, atol=TOL) for i in range(m))

    if pivs == n-1:
        return "SPD", "Sistema possível e determinado (SPD)"

    return "SPI", "Sistema possível e indeterminado (SPI)"

def construir_solucao_array(A_str):
    """
    Constrói a solução a partir da matriz normalizada em formato de string.
    Retorna uma tupla: (solução textual, solução numérica)
    A solução numérica é uma lista de floats para SPD e lista de Nones para SPI.
    """
    m, n = A_str.shape
    sol_textual = np.empty(n - 1, dtype=object)
    sol_numerica_list = [None] * (n - 1)
    livres = set(range(n - 1))

    for i in range(m):
        pivot_found = False
        for j in range(n - 1):
            if A_str[i, j] in ("1", "1/1"):
                livres.discard(j)

                expr = []
                const_term_str = A_str[i, -1]
                const_term_frac = Fraction(const_term_str)
                is_spd_row = True

                for k in range(n - 1):
                    if k == j: continue

                    cc_str = A_str[i, k]
                    if cc_str not in ("0", "0/1"):
                        is_spd_row = False
                        val_cc = Fraction(cc_str)

                        sign_term = Fraction(-1) * val_cc
                        sign = "+" if sign_term > 0 else "-"
                        abs_val_cc = abs(sign_term)

                        if abs_val_cc == 1:
                            term = f"{var_label(k)}"
                        else:
                            term = f"{abs_val_cc}{var_label(k)}"

                        expr.append(f"{sign} {term}")

                sol_expr = " ".join(expr)

                if const_term_str in ("0", "0/1") and not sol_expr:
                    sol_textual[j] = "0"
                else:
                    const_part = f"{const_term_str}" if const_term_str not in ("0", "0/1") else ""
                    sol_textual[j] = f"{const_part} {sol_expr}".strip()

                if is_spd_row and not expr:
                    sol_numerica_list[j] = float(const_term_frac)

                pivot_found = True
                break

    for j in livres:
        sol_textual[j] = f"{var_label(j)} (livre)"
        sol_numerica_list[j] = None

    return sol_textual, sol_numerica_list

def Resolva(A, verbose=True):
    """
    Resolve um sistema de equações lineares passo a passo.
    Retorna a solução textual, a descrição, os passos e a solução numérica (se for SPD).
    """
    steps_data = []
    Af = to_float_array(A.copy())
    Reduzir(Af, steps_data, verbose)
    Af = remover_linhas_nulas(Af, steps_data, verbose)

    cls, desc = classificar_sistema(Af)

    if cls == "SI":
        if verbose:
            print(f"❌ {desc}")
        return np.array([]), desc, steps_data, None

    eliminar_acima_com_pivo(Af, steps_data, verbose)

    An_str, An_float = Normalizar(Af)

    steps_data.append({
        'matriz_str': An_str,
        'matriz_float': An_float,
        'step_name': "Matriz Normalizada"
    })

    if verbose:
        print("=== Normalizada ===")
        print_matriz(An_str)

    sol_textual, sol_numerica_list = construir_solucao_array(An_str)

    solution_point = None
    if cls == "SPD":
        if all(val is not None for val in sol_numerica_list):
            solution_point = np.array(sol_numerica_list, dtype=float)
        else:
            try:
                solution_point = Af[:,-1]
                if len(solution_point) < 3:
                     solution_point = np.concatenate((solution_point, [0]))
                elif len(solution_point) > 3:
                    solution_point = solution_point[:3]
                solution_point = solution_point.astype(float)
            except Exception as e:
                print(f"Alerta: Falha ao extrair solução numérica para SPD: {e}")
                solution_point = None

    if verbose:
        print("SOLUÇÃO:")
        for i, s in enumerate(sol_textual, 1):
            print(f"x_{i} = {s}")

    return sol_textual, desc, steps_data, solution_point

def resolver_sistema_linear_passo_a_passo(matriz, verbose=False):
    """Função wrapper para Resolva, adaptada para o Streamlit."""
    sol, desc, steps_data, solution_point = Resolva(matriz, verbose)

    if sol is None:
        sol = np.array([], dtype=object)

    return desc, sol, steps_data, solution_point

# =============================================================================
# CLI PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    print("RESOLVEDOR DE SISTEMAS LINEARES 3D")
    entrada = input("> ").strip()

    try:
        matriz_usuario = ast.literal_eval(entrada)
        M = validar_matriz_usuario(matriz_usuario)

        desc, sol, steps_data, solution_point = resolver_sistema_linear_passo_a_passo(M, verbose=True)

        print(f"Resultado: {desc}")
        print("Solução:", sol)
        if solution_point is not None:
            print("Ponto de Solução Numérica:", solution_point)

        for i, step_info in enumerate(steps_data):
            print(f"\n--- Passo {i+1}: {step_info['step_name']} ---")

            if 'matriz_str' in step_info:
                print_matriz(step_info['matriz_str'])
            else:
                print_matriz(step_info['matriz'])

    except (ValueError, SyntaxError) as e:
        print(f"Erro na leitura da entrada: {e}")
        print("Formato esperado: [[a1,b1,c1,d1],[a2,b2,c2,d2],[a3,b3,c3,d3]] com valores numéricos.")

# =============================================================================
# FUNÇÕES AUXILIARES PARA INTERFACE
# =============================================================================

def print_matriz(A):
    """Imprime matriz de forma formatada"""
    if A.size == 0:
        print("Matriz vazia")
        return

    if A.dtype == object:
        for i, row in enumerate(A):
            linha = " ".join(f"{str(val):>8}" for val in row)
            print(f"[{linha}]")
    else:
        for i, row in enumerate(A):
            linha = " ".join(f"{val:8.3f}" for val in row)
            print(f"[{linha}]")
    print()
