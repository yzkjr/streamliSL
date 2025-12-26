"""
modules/graphics.py
Responsável pela geração de gráficos 2D e 3D utilizando Plotly.
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# =============================================================================
# CONFIGURAÇÕES DE VISUALIZAÇÃO
# =============================================================================

PLOT_CONFIG = {
    "mesh_range": (-12, 12),
    "n_points": 80,
    "tolerance": 1e-10,
    "opacity": 0.6,
    "colors": px.colors.qualitative.Plotly,
    "fig_size": {"width": 900, "height": 700},
}

# =============================================================================
# FUNÇÕES INTERNAS (AUXILIARES)
# =============================================================================

def _build_meshes(vals):
    """
    Cria 3 grades independentes para (x,y), (y,z) e (x,z).
    Evita distorções ao isolar variáveis com coeficientes zerados.
    """
    xx_xy, yy_xy = np.meshgrid(vals, vals)
    yy_yz, zz_yz = np.meshgrid(vals, vals)
    xx_xz, zz_xz = np.meshgrid(vals, vals)

    return {
        "xy": (xx_xy, yy_xy),
        "yz": (yy_yz, zz_yz),
        "xz": (xx_xz, zz_xz),
    }

def _calculate_plane_mesh(equation, meshes, tol):
    """
    Retorna malha {x,y,z} para o plano ax + by + cz = d, escolhendo automaticamente
    qual variável isolar (preferindo a que tem coeficiente "maior" em módulo).
    """
    a, b, c, d = equation

    # Plano degenerado
    if np.allclose([a, b, c], 0.0, atol=tol):
        return None

    abs_coeffs = np.array([abs(a), abs(b), abs(c)])
    idx = int(np.argmax(abs_coeffs))

    # Isola Z
    if idx == 2 and abs(c) > tol:
        xx, yy = meshes["xy"]
        zz = (d - a * xx - b * yy) / c
        return {"x": xx, "y": yy, "z": zz}

    # Isola Y
    if idx == 1 and abs(b) > tol:
        xx, zz = meshes["xz"]
        yy = (d - a * xx - c * zz) / b
        return {"x": xx, "y": yy, "z": zz}

    # Isola X
    if idx == 0 and abs(a) > tol:
        yy, zz = meshes["yz"]
        xx = (d - b * yy - c * zz) / a
        return {"x": xx, "y": yy, "z": zz}

    # Fallback
    if abs(c) > tol:
        xx, yy = meshes["xy"]
        zz = (d - a * xx - b * yy) / c
        return {"x": xx, "y": yy, "z": zz}
    return None


def _add_surface_legend_entry(fig, color, name):
    """Adiciona um Scatter3d 'invisível' para criar legenda de superfícies."""
    fig.add_trace(
        go.Scatter3d(
            x=[None], y=[None], z=[None],
            mode="markers",
            marker=dict(size=8, color=color),
            name=name,
            showlegend=True,
        )
    )


# =============================================================================
# FUNÇÕES DE PLOTAGEM (API PÚBLICA)
# =============================================================================

def plotar_retas_2d(matriz, step=None, solution_point=None):
    """Plota retas em 2D para sistemas 2x2."""
    fig = go.Figure()
    x_range = np.linspace(-10, 10, 200)

    matriz = np.array(matriz, dtype=float)

    for i, row in enumerate(matriz):
        a, b, d = row
        color = PLOT_CONFIG["colors"][i % len(PLOT_CONFIG["colors"])]

        if abs(b) > PLOT_CONFIG["tolerance"]:
            y = (d - a * x_range) / b
            fig.add_trace(
                go.Scatter(
                    x=x_range,
                    y=y,
                    mode="lines",
                    name=f"Eq {i+1}: {a:.3g}x + {b:.3g}y = {d:.3g}",
                    line=dict(color=color, width=3),
                )
            )
        elif abs(a) > PLOT_CONFIG["tolerance"]:
            x_val = d / a
            fig.add_shape(
                type="line",
                x0=x_val, y0=-10, x1=x_val, y1=10,
                line=dict(color=color, width=3, dash="dash"),
            )
            fig.add_trace(
                go.Scatter(
                    x=[x_val], y=[0],
                    mode="lines",
                    name=f"Eq {i+1}: {a:.3g}x = {d:.3g} (vertical)",
                    line=dict(color=color, width=3),
                )
            )

    if solution_point is not None and len(solution_point) >= 2:
        fig.add_trace(
            go.Scatter(
                x=[solution_point[0]],
                y=[solution_point[1]],
                mode="markers",
                marker=dict(size=12, color="red", symbol="x"),
                name=f"Solução ({solution_point[0]:.2f}, {solution_point[1]:.2f})",
            )
        )

    fig.update_layout(
        title=step if step else "Visualização 2D",
        xaxis=dict(title="Eixo X", range=[-10, 10], zeroline=True, zerolinewidth=2),
        yaxis=dict(title="Eixo Y", range=[-10, 10], zeroline=True, zerolinewidth=2),
        width=700,
        height=600,
        template="plotly_dark",
        legend=dict(itemsizing="constant"),
    )
    return fig


def plotar_planos_3d(matriz, step=None, config=None, solution_point=None):
    """Plota planos em 3D para sistemas 3x3."""
    cfg = PLOT_CONFIG if config is None else config

    matriz = np.array(matriz, dtype=float)
    m, n = matriz.shape
    if n != 4:
        return None

    vals = np.linspace(cfg["mesh_range"][0], cfg["mesh_range"][1], cfg["n_points"])
    meshes = _build_meshes(vals)

    fig = go.Figure()

    for i, (a, b, c, d) in enumerate(matriz):
        if np.allclose([a, b, c], 0.0, atol=cfg["tolerance"]):
            continue

        mesh = _calculate_plane_mesh((a, b, c, d), meshes, cfg["tolerance"])
        if mesh is None:
            continue

        color = cfg["colors"][i % len(cfg["colors"])]
        name = f"Plano {i+1}: {a:.3g}x + {b:.3g}y + {c:.3g}z = {d:.3g}"

        fig.add_trace(
            go.Surface(
                x=mesh["x"], y=mesh["y"], z=mesh["z"],
                opacity=cfg["opacity"],
                colorscale=[[0, color], [1, color]],
                showscale=False,
                hoverinfo="skip",
                showlegend=False, 
            )
        )
        _add_surface_legend_entry(fig, color, name)

    if (
        solution_point is not None
        and len(solution_point) == 3
        and all(isinstance(val, (int, float)) for val in solution_point)
    ):
        x_sol, y_sol, z_sol = solution_point
        fig.add_trace(
            go.Scatter3d(
                x=[x_sol], y=[y_sol], z=[z_sol],
                mode="markers+text",
                text=[f"Solução ({x_sol:.2f}, {y_sol:.2f}, {z_sol:.2f})"],
                textposition="top center",
                marker=dict(size=8, color="gold", symbol="circle", line=dict(width=1)),
                name="Solução",
                hovertemplate="Solução<br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        title_text=f"Visualização 3D: {step}" if step else "Visualização 3D",
        width=cfg["fig_size"]["width"],
        height=cfg["fig_size"]["height"],
        template="plotly_dark",
        scene=dict(
            aspectmode="cube",
            xaxis=dict(title="x", nticks=10),
            yaxis=dict(title="y", nticks=10),
            zaxis=dict(title="z", nticks=10),
        ),
        legend=dict(itemsizing="constant"),
    )

    return fig
