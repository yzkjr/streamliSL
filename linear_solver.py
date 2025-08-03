
"""
=============================================================================
RESOLVEDOR DE SISTEMAS LINEARES 3D - MÓDULO PRINCIPAL
=============================================================================

Este módulo implementa um resolvedor completo de sistemas de equações lineares
3x3 usando eliminação gaussiana com visualização 3D interativa.

Funcionalidades principais:
- Eliminação gaussiana com pivotamento
- Eliminação regressiva (Gauss)
- Classificação de sistemas (SPD, SPI, SI)
- Visualização 3D dos planos geométricos

Autor: Izaac Soares, Jeferson Danilo
Versão: 1.0
Dependências: numpy, plotly, fractions
"""

import numpy as np
from fractions import Fraction
import ast
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# CONFIGURAÇÕES GLOBAIS E CONSTANTES
# =============================================================================

# Configurações para visualização 3D dos planos
PLOT_CONFIG = {
    'mesh_range': (-12, 12),           # Intervalo de valores para o grid 3D
    'n_points': 80,                    # Resolução do grid (80x80 pontos)
    'tolerance': 1e-10,                # Tolerância para cálculos numéricos
    'opacity': 0.6,                    # Transparência dos planos (0-1)
    'colors': px.colors.qualitative.Plotly,  # Paleta de cores para os planos
    'background_color': 'white',       # Cor de fundo do gráfico
    'fig_size': {'width': 900, 'height': 700},  # Dimensões da figura
    'tick_step': 1,                    # Espaçamento entre ticks dos eixos
    'axis_line_width': 3,              # Espessura das linhas dos eixos
    'cone_size': 1.0,                  # Tamanho dos cones direcionais
    'label_offset': 1.0,               # Offset para posicionamento de labels
}

# Tolerância global para comparações numéricas
TOL = 1e-10

# =============================================================================
# VALIDAÇÃO DE ENTRADA DE DADOS
# =============================================================================

def validar_coeficientes(*args):
    """
    Valida se todos os coeficientes fornecidos são números válidos.
    
    Args:
        *args: Coeficientes a serem validados (int ou float)
        
    Returns:
        bool: True se todos os coeficientes são válidos
        
    Raises:
        ValueError: Se algum coeficiente não for um número
        
    Exemplo:
        >>> validar_coeficientes(1, 2.5, -3, 4.0)
        True
        >>> validar_coeficientes(1, "a", 3)  # Levanta ValueError
    """
    for i, val in enumerate(args):
        if not isinstance(val, (int, float)):
            raise ValueError(f"Coeficiente {i+1} inválido: {val}. Deve ser um número.")
    return True


def validar_matriz_usuario(matriz):
    """
    Valida e converte a matriz de entrada do usuário para o formato padrão.
    
    A matriz deve ter formato [[a1,b1,c1,d1], [a2,b2,c2,d2], [a3,b3,c3,d3]]
    onde cada linha representa uma equação linear: ax + by + cz = d
    
    Args:
        matriz (list, tuple, np.ndarray): Matriz de coeficientes de entrada
        
    Returns:
        np.ndarray: Matriz validada como array numpy de floats
        
    Raises:
        ValueError: Se a matriz não atender aos critérios de validação
        
    Exemplo:
        >>> matriz = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]
        >>> validar_matriz_usuario(matriz)
        array([[ 1.,  2.,  3.,  4.],
               [ 5.,  6.,  7.,  8.],
               [ 9., 10., 11., 12.]])
    """
    # Verifica o tipo de entrada
    if not isinstance(matriz, (list, tuple, np.ndarray)):
        raise ValueError("A entrada deve ser uma lista/tuple/array.")
    
    # Converte para array numpy
    try:
        arr = np.array(matriz, dtype=float)
    except Exception as e:
        raise ValueError(f"Erro ao processar matriz: {e}")

    # Valida dimensões
    if arr.ndim != 2:
        raise ValueError("A matriz deve ser bidimensional.")
    
    m, n = arr.shape
    if n != 4:
        raise ValueError(f"A matriz deve ter 4 colunas (a, b, c, d). Recebido: {n}")
    if m == 0:
        raise ValueError("A matriz não pode ser vazia.")
    
    return arr

# =============================================================================
# FUNÇÕES MATEMÁTICAS E UTILITÁRIOS
# =============================================================================

def MDC(a, b):
    """
    Calcula o Máximo Divisor Comum (MDC) de dois números inteiros.
    
    Implementa o algoritmo euclidiano clássico para encontrar o MDC.
    
    Args:
        a (int): Primeiro número
        b (int): Segundo número
        
    Returns:
        int: MDC dos dois números
        
    Exemplo:
        >>> MDC(48, 18)
        6
        >>> MDC(17, 13)
        1
    """
    a, b = abs(int(a)), abs(int(b))
    while b:
        a, b = b, a % b
    return a


def calcular_mdc_array(linha):
    """
    Calcula o MDC de todos os elementos não-zero de uma linha da matriz.
    
    Esta função é usada para simplificar linhas da matriz durante a eliminação,
    mantendo apenas coeficientes inteiros quando possível.
    
    Args:
        linha (np.ndarray): Linha da matriz para calcular o MDC
        
    Returns:
        int: MDC de todos os elementos não-zero da linha
        
    Exemplo:
        >>> linha = np.array([6, 9, 12, 15])
        >>> calcular_mdc_array(linha)
        3
    """
    elems = []
    for x in linha:
        if abs(x) > TOL:
            if np.isclose(x, round(x), atol=TOL):
                elems.append(int(round(x)))
            else:
                return 1  # Se há números não-inteiros, retorna 1
    
    if not elems:
        return 1
    
    # Calcula MDC sequencialmente
    m = elems[0]
    for e in elems[1:]:
        m = MDC(m, e)
    return abs(m)


def encontrar_primeiro_nao_zero(linha):
    """
    Encontra o primeiro elemento não-zero em uma linha da matriz (pivô).
    
    Args:
        linha (np.ndarray): Linha da matriz a ser analisada
        
    Returns:
        tuple: (valor_do_pivo, indice_do_pivo) ou (None, None) se não houver pivô
        
    Exemplo:
        >>> linha = np.array([0, 0, 3, 5])
        >>> encontrar_primeiro_nao_zero(linha)
        (3.0, 2)
    """
    for j, x in enumerate(linha):
        if abs(x) > TOL:
            return x, j
    return None, None


def to_float_array(A):
    """
    Converte uma matriz para array de floats sem criar cópia desnecessária.
    
    Args:
        A (np.ndarray): Matriz a ser convertida
        
    Returns:
        np.ndarray: Matriz como array de floats
    """
    return A.astype(float, copy=False)


def dividir_por_mdc(linha):
    """
    Divide uma linha da matriz pelo MDC de seus elementos para simplificação.
    
    Esta operação mantém a proporcionalidade da equação enquanto trabalha
    com números menores e mais simples.
    
    Args:
        linha (np.ndarray): Linha da matriz a ser simplificada
        
    Returns:
        np.ndarray: Linha simplificada
        
    Exemplo:
        >>> linha = np.array([6, 9, 12, 15])
        >>> dividir_por_mdc(linha)
        array([2., 3., 4., 5.])
    """
    m = calcular_mdc_array(linha)
    if m != 1:
        linha = linha / m
        # Arredonda valores muito próximos de inteiros
        linha = np.where(np.isclose(linha, np.round(linha), atol=TOL), 
                        np.round(linha), linha)
    return linha


def frac_str(num, den):
    """
    Converte uma fração numérica para representação em string simplificada.
    
    Args:
        num (float): Numerador da fração
        den (float): Denominador da fração
        
    Returns:
        str: Representação em string da fração simplificada
        
    Exemplo:
        >>> frac_str(3, 6)
        '1/2'
        >>> frac_str(5, 1)
        '5'
        >>> frac_str(7, 0)
        'Indefinido'
    """
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
    """
    Gera o rótulo padrão para uma variável baseado em seu índice.
    
    Args:
        index (int): Índice da variável (0-based)
        
    Returns:
        str: Rótulo da variável (x_1, x_2, x_3, etc.)
        
    Exemplo:
        >>> var_label(0)
        'x_1'
        >>> var_label(2)
        'x_3'
    """
    return f"x_{index + 1}"

# =============================================================================
# VISUALIZAÇÃO 3D DOS PLANOS
# =============================================================================

def _calculate_plane_mesh(equation, meshes, tol):
    """
    Calcula a malha 3D para renderização de um plano específico.
    
    Determina qual variável usar como função das outras duas, baseado
    em qual coeficiente tem maior magnitude (evita divisão por zero).
    
    Args:
        equation (tuple): Coeficientes (a, b, c, d) da equação ax+by+cz=d
        meshes (tuple): Malhas de coordenadas para os três planos de projeção
        tol (float): Tolerância para comparações numéricas
        
    Returns:
        dict or None: Dicionário com coordenadas {'x': xx, 'y': yy, 'z': zz}
                     ou None se o plano for degenerado
    """
    a, b, c, d = equation
    xx_xy, yy_xy, yy_yz, zz_yz, xx_xz, zz_xz = meshes
    
    # Prioriza o coeficiente de maior magnitude para estabilidade numérica
    if abs(c) > tol:  # Resolve para z em função de x e y
        zz = (d - a*xx_xy - b*yy_xy) / c
        return {'x': xx_xy, 'y': yy_xy, 'z': zz}
    elif abs(b) > tol:  # Resolve para y em função de x e z
        yy = (d - a*xx_xz - c*zz_xz) / b
        return {'x': xx_xz, 'y': yy, 'z': zz_xz}
    elif abs(a) > tol:  # Resolve para x em função de y e z
        xx = (d - b*yy_yz - c*zz_yz) / a
        return {'x': xx, 'y': yy_yz, 'z': zz_yz}
    
    return None  # Plano degenerado (todos os coeficientes são zero)


def plotar_planos_3d(matriz, step=None, config=None, solution_point=None):
    """
    Gera visualização 3D interativa dos planos representados pela matriz.
    
    Cada linha da matriz representa um plano no espaço 3D. A função
    renderiza todos os planos simultaneamente com cores distintas e
    opcionalmente marca o ponto de solução.
    
    Args:
        matriz (np.ndarray): Matriz de coeficientes [a, b, c, d] por linha
        step (str, optional): Descrição do passo atual para o título
        config (dict, optional): Configurações de plotagem personalizadas
        solution_point (list, optional): Coordenadas [x, y, z] da solução
        
    Returns:
        plotly.graph_objects.Figure: Figura Plotly interativa
        
    Exemplo:
        >>> matriz = np.array([[1, 1, 1, 6], [1, -1, 2, 5], [1, -1, -3, -10]])
        >>> fig = plotar_planos_3d(matriz, "Sistema Original")
        >>> fig.show()  # Exibe a visualização interativa
    """
    # Usa configuração padrão se não fornecida
    if config is None:
        config = PLOT_CONFIG
    
    # Converte entrada para array numpy se necessário
    if not isinstance(matriz, np.ndarray):
        matriz = np.array(matriz, dtype=float)
    
    m, n = matriz.shape
    if n != 4:
        return None  # Matriz inválida
    
    cfg = config
    
    # Gera grid de coordenadas para renderização
    vals = np.linspace(cfg['mesh_range'][0], cfg['mesh_range'][1], cfg['n_points'])
    meshes = np.meshgrid(vals, vals)
    all_meshes = (meshes[0], meshes[1], meshes[0], meshes[1], meshes[0], meshes[1])
    
    # Inicializa figura Plotly
    fig = go.Figure()
    
    # Renderiza cada plano da matriz
    for i, (a, b, c, d) in enumerate(matriz):
        # Pula planos degenerados (todos os coeficientes zero)
        if np.allclose([a, b, c], 0, atol=cfg['tolerance']):
            continue
        
        # Calcula malha do plano
        mesh = _calculate_plane_mesh((a, b, c, d), all_meshes, cfg['tolerance'])
        if mesh is None:
            continue
        
        # Seleciona cor do plano
        color = cfg['colors'][i % len(cfg['colors'])]
        
        # Adiciona superfície do plano ao gráfico
        fig.add_trace(go.Surface(
            x=mesh['x'], y=mesh['y'], z=mesh['z'],
            opacity=cfg['opacity'],
            colorscale=[[0, color], [1, color]],
            showscale=False,
            name=f"Plano {i+1}: {a:.1f}x + {b:.1f}y + {c:.1f}z = {d:.1f}"
        ))
    
    # Adiciona ponto de solução se fornecido
    if (solution_point is not None and 
        len(solution_point) == 3 and 
        all(isinstance(val, (int, float)) for val in solution_point)):
        
        x_sol, y_sol, z_sol = solution_point
        
        # Marca o ponto de solução
        fig.add_trace(go.Scatter3d(
            x=[x_sol], y=[y_sol], z=[z_sol],
            mode='markers',
            marker=dict(
                size=10, color='gold', symbol='circle',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name=f"Solução: ({x_sol:.2f}, {y_sol:.2f}, {z_sol:.2f})",
            hoverinfo='name+x+y+z'
        ))
        
        # Adiciona anotação do ponto de solução
        fig.update_layout(
            scene=dict(
                annotations=[
                    dict(
                        x=x_sol, y=y_sol, z=z_sol,
                        text=f"Solução: ({x_sol:.1f}, {y_sol:.1f}, {z_sol:.1f})",
                        showarrow=True, arrowhead=1, ax=20, ay=-20,
                        font=dict(size=12, color="white"),
                        bgcolor="rgba(0,0,0,0.5)"
                    )
                ]
            )
        )
    
    # Configura layout da figura
    fig.update_layout(
        title_text=f"Visualização 3D: {step}" if step else "Visualização 3D",
        width=cfg['fig_size']['width'],
        height=cfg['fig_size']['height'],
        scene=dict(
            aspectmode='cube',  # Mantém proporções iguais nos eixos
            xaxis=dict(tickfont=dict(size=14), nticks=10),
            yaxis=dict(tickfont=dict(size=14), nticks=10),
            zaxis=dict(tickfont=dict(size=14), nticks=10)
        )
    )
    
    return fig

# =============================================================================
# ALGORITMOS DE ELIMINAÇÃO GAUSSIANA
# =============================================================================

def Eliminar(A, i, j, cnt, steps_data, verbose=True):
    """
    Executa uma operação elementar de eliminação em uma linha específica.
    
    Esta função implementa a operação L_i = L_i - (A[i,j]/A[pivot,j]) * L_pivot
    para zerar o elemento A[i,j], onde pivot é a linha j-cnt.
    
    Args:
        A (np.ndarray): Matriz aumentada do sistema
        i (int): Índice da linha a ser modificada
        j (int): Índice da coluna do elemento a ser zerado
        cnt (int): Contador de colunas processadas
        steps_data (list): Lista para armazenar passos da resolução
        verbose (bool): Se True, imprime matriz após operação
        
    Efeitos colaterais:
        - Modifica a matriz A in-place
        - Adiciona passo aos steps_data
        - Imprime matriz se verbose=True
    """
    A_copy = A.copy()
    elem = A_copy[i, j]          # Elemento a ser zerado
    p = A_copy[j - cnt, j]       # Elemento pivô
    
    if abs(elem) < TOL:
        return  # Elemento já é efetivamente zero
    
    # Calcula MDC para simplificar a operação
    g = abs(MDC(p, elem))
    
    # Constrói string descritiva da operação
    operation_str = ""
    if g != 1:
        # Operação com frações simplificadas
        A_copy[i] = (p/g)*A_copy[i] - (elem/g)*A_copy[j - cnt]
        num_str = frac_str(elem, g)
        den_str = frac_str(p, g)
        if den_str == "1":
            operation_str = f"L{i+1} - {num_str}L{j-cnt+1}"
        else:
            operation_str = f"L{i+1} - ({num_str}/{den_str})L{j-cnt+1}"
    else:
        # Operação com números inteiros
        A_copy[i] = p*A_copy[i] - elem*A_copy[j - cnt]
        if abs(p) > TOL:
            fraction_val = elem / p
            if np.isclose(fraction_val, 1.0, atol=TOL):
                operation_str = f"L{i+1} - L{j-cnt+1}"
            elif np.isclose(fraction_val, -1.0, atol=TOL):
                operation_str = f"L{i+1} + L{j-cnt+1}"
            else:
                operation_str = f"L{i+1} - ({elem:.1f}/{p:.1f})L{j-cnt+1}"
        else:
            operation_str = f"L{i+1} - ({elem:.1f}/{p:.1f})L{j-cnt+1}"
    
    # Simplifica a linha resultante
    A_copy[i] = dividir_por_mdc(A_copy[i])
    
    # Registra o passo
    if verbose:
        print_matriz(A_copy)
    
    steps_data.append({
        'matriz': A_copy.copy(), 
        'step_name': f"Eliminação: {operation_str}"
    })
    
    # Atualiza matriz original
    A[:] = A_copy[:]


def Reduzir(A, steps_data, verbose=True):
    """
    Executa a fase progressiva da eliminação gaussiana.
    
    Transforma a matriz em forma escalonada através de:
    1. Pivotamento (troca de linhas quando necessário)
    2. Eliminação de elementos abaixo da diagonal principal
    
    Args:
        A (np.ndarray): Matriz aumentada do sistema
        steps_data (list): Lista para armazenar passos da resolução
        verbose (bool): Se True, imprime informações debug
        
    Efeitos colaterais:
        - Modifica A para forma escalonada
        - Adiciona múltiplos passos aos steps_data
    """
    A_copy = A.copy()
    m, n = A_copy.shape
    cnt = 0  # Contador de colunas "perdidas" (sem pivô)
    
    if verbose:
        print("=== ELIMINAÇÃO GAUSSIANA PROGRESSIVA ===")
        print_matriz(A_copy)
    
    steps_data.append({
        'matriz': A_copy.copy(), 
        'step_name': "Início da Eliminação Gaussiana"
    })
    
    # Processa cada coluna
    for j in range(n):
        if j - cnt >= m:
            break  # Mais colunas que linhas restantes
        
        # Verifica se precisa de pivotamento
        if abs(A_copy[j-cnt, j]) < TOL:
            # Procura linha não-zero abaixo para trocar
            for i in range(j+1-cnt, m):
                if abs(A_copy[i, j]) > TOL:
                    A_copy[[j-cnt, i]] = A_copy[[i, j-cnt]]  # Troca linhas
                    if verbose:
                        print_matriz(A_copy)
                    steps_data.append({
                        'matriz': A_copy.copy(), 
                        'step_name': f"Permutação: L{j-cnt+1} <-> L{i+1}"
                    })
                    break
            else:
                # Nenhuma linha não-zero encontrada, pula esta coluna
                cnt += 1
                continue
        
        # Elimina elementos abaixo do pivô
        for i in range(j+1-cnt, m):
            Eliminar(A_copy, i, j, cnt, steps_data, verbose)
    
    if verbose:
        print("=== MATRIZ ESCALONADA ===")
        print_matriz(A_copy)
    
    steps_data.append({
        'matriz': A_copy.copy(), 
        'step_name': "Matriz Escalonada"
    })
    
    A[:] = A_copy[:]


def remover_linhas_nulas(A, steps_data, verbose=True):
    """
    Remove linhas compostas inteiramente por zeros da matriz.
    
    Linhas nulas não contribuem para a solução e podem ser removidas
    para simplificar análises subsequentes.
    
    Args:
        A (np.ndarray): Matriz escalonada
        steps_data (list): Lista para armazenar passos
        verbose (bool): Se True, imprime informações debug
        
    Returns:
        np.ndarray: Nova matriz sem linhas nulas
    """
    A_copy = A.copy()
    
    # Filtra linhas não-nulas
    keep = [row for row in A_copy if not np.allclose(row, 0, atol=TOL)]
    B = np.array(keep, dtype=float)
    
    if verbose:
        print("=== REMOÇÃO DE LINHAS NULAS ===")
        print_matriz(B)
    
    steps_data.append({
        'matriz': B.copy(), 
        'step_name': "Remoção de Linhas Nulas"
    })
    
    return B


def verificar_linhas_inconsistentes(A):
    """
    Verifica se existem linhas que indicam inconsistência no sistema.
    
    Uma linha inconsistente tem a forma [0, 0, 0, d] onde d ≠ 0,
    o que representa a equação impossível 0 = d.
    
    Args:
        A (np.ndarray): Matriz escalonada
        
    Returns:
        bool: True se há inconsistência, False caso contrário
    """
    for row in A:
        if np.allclose(row[:-1], 0, atol=TOL) and abs(row[-1]) > TOL:
            return True
    return False


def eliminar_acima_com_pivo(A, steps_data, verbose=True):
    """
    Executa a fase regressiva da eliminação (eliminação de Gauss-Jordan).
    
    Elimina elementos acima da diagonal principal, trabalhando de baixo
    para cima para obter a forma escalonada reduzida.
    
    Args:
        A (np.ndarray): Matriz escalonada
        steps_data (list): Lista para armazenar passos
        verbose (bool): Se True, imprime informações debug
        
    Efeitos colaterais:
        - Modifica A para forma escalonada reduzida
        - Adiciona passos aos steps_data
    """
    A_copy = A.copy()
    m, _ = A_copy.shape
    
    if verbose:
        print("=== ELIMINAÇÃO REGRESSIVA ===")
    
    # Processa linhas de baixo para cima
    for i in range(m-1, -1, -1):
        pv, pc = encontrar_primeiro_nao_zero(A_copy[i])
        
        if pv is None:
            continue  # Linha nula, pula
        
        # Simplifica a linha atual
        A_copy[i] = dividir_por_mdc(A_copy[i])
        p = A_copy[i, pc]  # Elemento pivô
        
        # Elimina elementos acima do pivô
        for j in range(i-1, -1, -1):
            e = A_copy[j, pc]  # Elemento a ser zerado
            
            if abs(e) < TOL:
                continue  # Já é zero
            
            # Calcula MDC para simplificar operação
            g = abs(MDC(p, e))
            
            # Constrói string descritiva
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
                    operation_str = f"L{j+1} - ({e:.1f}/{p:.1f})L{i+1}"
            
            # Simplifica linha resultante
            A_copy[j] = dividir_por_mdc(A_copy[j])
            
            if verbose:
                print_matriz(A_copy)
            
            steps_data.append({
                'matriz': A_copy.copy(), 
                'step_name': f"Eliminação: {operation_str}"
            })
    
    A[:] = A_copy[:]


def Normalizar(A):
    """
    Normaliza a matriz para a forma escalonada reduzida canônica.
    
    Divide cada linha pelo seu elemento pivô para que todos os pivôs
    sejam igual a 1. Retorna duas versões: uma com strings para exibição
    e outra com floats para cálculos.
    
    Args:
        A (np.ndarray): Matriz escalonada reduzida
        
    Returns:
        tuple: (matriz_strings, matriz_floats)
            - matriz_strings: Para exibição de frações exatas
            - matriz_floats: Para cálculos numéricos e plotagem
    """
    if A.size == 0:
        return np.array([], dtype=object), np.array([], dtype=float)
    
    A_copy = A.copy()
    m, n = A_copy.shape
    R_str = np.zeros((m, n), dtype=object)    # Versão string
    R_float = np.zeros((m, n), dtype=float)   # Versão float
    
    for i in range(m):
        pv, pc = encontrar_primeiro_nao_zero(A_copy[i])
        
        if pv is None:
            # Linha nula
            R_str[i, :] = "0"
            R_float[i, :] = 0.0
            continue
        
        # Simplifica a linha
        L = dividir_por_mdc(A_copy[i])
        pv = L[pc]  # Novo valor do pivô
        
        # Versão float (para plotagem e cálculos)
        L_float = L / pv
        L_float = np.where(np.isclose(L_float, np.round(L_float), atol=TOL), 
                          np.round(L_float), L_float)
        R_float[i] = L_float
        
        # Versão string (para exibição da solução)
        for j in range(n):
            v = L[j]
            
            if abs(v) < TOL:
                R_str[i, j] = "0"
            elif j == pc:
                R_str[i, j] = "1"  # Pivô normalizado
            else:
                R_str[i, j] = frac_str(v, pv)
    
    return R_str, R_float

# =============================================================================
# CLASSIFICAÇÃO E CONSTRUÇÃO DE SOLUÇÕES
# =============================================================================

def classificar_sistema(A):
    """
    Classifica o sistema linear baseado na matriz escalonada.
    
    Analisa a matriz reduzida para determinar:
    - SI (Sistema Impossível): Há contradições
    - SPD (Sistema Possível Determinado): Solução única
    - SPI (Sistema Possível Indeterminado): Infinitas soluções
    
    Args:
        A (np.ndarray): Matriz escalonada reduzida
        
    Returns:
        tuple: (codigo, descricao)
            - codigo: "SI", "SPD", ou "SPI"
            - descricao: Descrição textual da classificação
    """
    m, n = A.shape
    
    # Verifica inconsistência (linha tipo [0, 0, 0, d] com d ≠ 0)
    if verificar_linhas_inconsistentes(A):
        return "SI", "Sistema impossível (SI)"
    
    # Conta número de pivôs (linhas não-nulas)
    pivs = sum(not np.allclose(A[i, :-1], 0, atol=TOL) for i in range(m))
    
    # Se número de pivôs = número de variáveis, sistema determinado
    if pivs == n-1:
        return "SPD", "Sistema possível e determinado (SPD)"
    
    # Caso contrário, sistema indeterminado
    return "SPI", "Sistema possível e indeterminado (SPI)"


def construir_solucao_array(A_str):
    """
    Constrói a solução do sistema a partir da matriz normalizada.
    
    Analisa a matriz escalonada reduzida para determinar as expressões
    das variáveis dependentes em função das variáveis livres.
    
    Args:
        A_str (np.ndarray): Matriz normalizada em formato string
        
    Returns:
        tuple: (solucao_textual, solucao_numerica)
            - solucao_textual: Array com expressões das variáveis
            - solucao_numerica: Lista com valores numéricos (SPD) ou None (SPI)
    """
    m, n = A_str.shape
    sol_textual = np.empty(n - 1, dtype=object)
    sol_numerica_list = [None] * (n - 1)
    livres = set(range(n - 1))  # Inicialmente todas são livres
    
    # Processa cada linha da matriz
    for i in range(m):
        # Procura pivô na linha
        for j in range(n - 1):
            if A_str[i, j] in ("1", "1/1"):
                livres.discard(j)  # Esta variável não é livre
                
                # Constrói expressão da variável dependente
                expr = []
                const_term_str = A_str[i, -1]
                const_term_frac = Fraction(const_term_str)
                is_spd_row = True  # Indica se esta linha contribui para SPD
                
                # Processa outros termos da equação
                for k in range(n - 1):
                    if k == j: continue  # Pula a própria variável
                    
                    cc_str = A_str[i, k]
                    if cc_str not in ("0", "0/1"):
                        is_spd_row = False  # Há variáveis livres
                        val_cc = Fraction(cc_str)
                        
                        # Calcula coeficiente com sinal oposto
                        sign_term = Fraction(-1) * val_cc
                        sign = "+" if sign_term > 0 else "-"
                        abs_val_cc = abs(sign_term)
                        
                        # Formata termo
                        if abs_val_cc == 1:
                            term = f"{var_label(k)}"
                        else:
                            term = f"{abs_val_cc}{var_label(k)}"
                        
                        expr.append(f"{sign} {term}")
                
                # Monta expressão final
                sol_expr = " ".join(expr)
                
                if const_term_str in ("0", "0/1") and not sol_expr:
                    sol_textual[j] = "0"
                else:
                    const_part = f"{const_term_str}" if const_term_str not in ("0", "0/1") else ""
                    sol_textual[j] = f"{const_part} {sol_expr}".strip()
                
                # Se for SPD, armazena valor numérico
                if is_spd_row and not expr:
                    sol_numerica_list[j] = float(const_term_frac)
                
                break  # Pivô encontrado, próxima linha
    
    # Marca variáveis livres
    for j in livres:
        sol_textual[j] = f"{var_label(j)} (livre)"
        sol_numerica_list[j] = None
    
    return sol_textual, sol_numerica_list

# =============================================================================
# FUNÇÃO PRINCIPAL DE RESOLUÇÃO
# =============================================================================

def Resolva(A, verbose=True):
    """
    Resolve completamente um sistema de equações lineares.
    
    Executa todo o processo de resolução:
    1. Eliminação gaussiana progressiva
    2. Remoção de linhas nulas
    3. Classificação do sistema
    4. Eliminação regressiva (se aplicável)
    5. Normalização da matriz
    6. Construção da solução
    
    Args:
        A (np.ndarray): Matriz aumentada do sistema
        verbose (bool): Se True, imprime informações debug
        
    Returns:
        tuple: (solucao_textual, descricao, passos, ponto_solucao)
            - solucao_textual: Array com expressões das variáveis
            - descricao: Classificação do sistema
            - passos: Lista de passos da resolução
            - ponto_solucao: Coordenadas numéricas (SPD) ou None
    """
    steps_data = []
    Af = to_float_array(A.copy())
    
    # Fase 1: Eliminação progressiva
    Reduzir(Af, steps_data, verbose)
    
    # Fase 2: Limpeza da matriz
    Af = remover_linhas_nulas(Af, steps_data, verbose)
    
    # Fase 3: Classificação
    cls, desc = classificar_sistema(Af)
    
    if cls == "SI":
        if verbose:
            print(f"❌ {desc}")
        return np.array([]), desc, steps_data, None
    
    # Fase 4: Eliminação regressiva
    eliminar_acima_com_pivo(Af, steps_data, verbose)
    
    # Fase 5: Normalização
    An_str, An_float = Normalizar(Af)
    
    steps_data.append({
        'matriz_str': An_str,
        'matriz_float': An_float,
        'step_name': "Matriz Normalizada"
    })
    
    if verbose:
        print("=== MATRIZ NORMALIZADA ===")
        print_matriz(An_str)
    
    # Fase 6: Construção da solução
    sol_textual, sol_numerica_list = construir_solucao_array(An_str)
    
    # Extrai ponto de solução para SPD
    solution_point = None
    if cls == "SPD":
        if all(val is not None for val in sol_numerica_list):
            solution_point = np.array(sol_numerica_list, dtype=float)
        else:
            # Fallback: usa última coluna da matriz
            try:
                solution_point = Af[:,-1]
                if len(solution_point) < 3:
                     solution_point = np.concatenate((solution_point, [0]))
                elif len(solution_point) > 3:
                    solution_point = solution_point[:3]
                solution_point = solution_point.astype(float)
            except Exception as e:
                if verbose:
                    print(f"Alerta: Falha ao extrair solução numérica para SPD: {e}")
                solution_point = None
    
    # Exibe solução
    if verbose:
        print("=== SOLUÇÃO FINAL ===")
        for i, s in enumerate(sol_textual, 1):
            print(f"x_{i} = {s}")
    
    return sol_textual, desc, steps_data, solution_point


def resolver_sistema_linear_passo_a_passo(matriz, verbose=False):
    """
    Função wrapper para integração com Streamlit.
    
    Interface simplificada da função Resolva() para uso em aplicações web.
    
    Args:
        matriz (np.ndarray): Matriz aumentada do sistema
        verbose (bool): Se True, imprime informações debug
        
    Returns:
        tuple: (descricao, solucao, passos, ponto_solucao)
    """
    sol, desc, steps_data, solution_point = Resolva(matriz, verbose)
    
    if sol is None:
        sol = np.array([], dtype=object)
    
    return desc, sol, steps_data, solution_point

# =============================================================================
# INTERFACE DE LINHA DE COMANDO
# =============================================================================

if __name__ == "__main__":
    """
    Interface de linha de comando para testes e uso direto.
    
    Permite inserir uma matriz manualmente e ver todo o processo de resolução.
    Formato esperado: [[a1,b1,c1,d1],[a2,b2,c2,d2],[a3,b3,c3,d3]]
    """
    print("="*60)
    print("RESOLVEDOR DE SISTEMAS LINEARES 3D")
    print("="*60)
    print("Formato de entrada: [[a1,b1,c1,d1],[a2,b2,c2,d2],[a3,b3,c3,d3]]")
    print("Onde cada linha representa: ax + by + cz = d")
    print("="*60)
    
    entrada = input("Digite a matriz do sistema: ").strip()
    
    try:
        # Parsing e validação da entrada
        matriz_usuario = ast.literal_eval(entrada)
        M = validar_matriz_usuario(matriz_usuario)
        
        print("\n" + "="*60)
        print("INICIANDO RESOLUÇÃO...")
        print("="*60)
        
        # Resolução completa
        desc, sol, steps_data, solution_point = resolver_sistema_linear_passo_a_passo(
            M, verbose=True
        )
        
        # Resultados finais
        print("\n" + "="*60)
        print("RESULTADO FINAL")
        print("="*60)
        print(f"Classificação: {desc}")
        print(f"Solução: {list(sol)}")
        
        if solution_point is not None:
            print(f"Ponto de Solução (x,y,z): {solution_point}")
        
        # Resumo dos passos
        print(f"\nTotal de passos executados: {len(steps_data)}")
        
    except (ValueError, SyntaxError) as e:
        print(f"\n❌ Erro na entrada: {e}")
        print("\nFormato esperado: [[a1,b1,c1,d1],[a2,b2,c2,d2],[a3,b3,c3,d3]]")
        print("Exemplo: [[1,1,1,6],[1,-1,2,5],[1,-1,-3,-10]]")

# =============================================================================
# FUNÇÕES AUXILIARES PARA INTERFACE
# =============================================================================

def print_matriz(A):
    """
    Imprime uma matriz de forma formatada e legível.
    
    Detecta automaticamente se a matriz contém strings ou números
    e aplica formatação apropriada.
    
    Args:
        A (np.ndarray): Matriz a ser impressa
        
    Exemplo:
        >>> A = np.array([[1.5, 2.0], [3.333, 4.0]])
        >>> print_matriz(A)
        [   1.500    2.000]
        [   3.333    4.000]
    """
    if A.size == 0:
        print("Matriz vazia")
        return
    
    if A.dtype == object:
        # Matriz de strings (frações)
        for i, row in enumerate(A):
            linha = " ".join(f"{str(val):>8}" for val in row)
            print(f"[{linha}]")
    else:
        # Matriz numérica
        for i, row in enumerate(A):
            linha = " ".join(f"{val:8.3f}" for val in row)
            print(f"[{linha}]")
    print()  # Linha em branco para separação

# =============================================================================
# FIM DO MÓDULO
# =============================================================================
