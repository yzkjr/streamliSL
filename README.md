# 📐 SistemaLinearLab

**SistemaLinearLab** é uma ferramenta educacional interativa desenvolvida para auxiliar estudantes e engenheiros na visualização e resolução de Sistemas de Equações Lineares.

O projeto utiliza o método de **Eliminação de Gauss-Jordan** para resolver sistemas de 2x2 até 5x5, oferecendo uma experiência visual rica com gráficos interativos e relatórios detalhados.

---

## ✨ Funcionalidades

* **Resolução Passo a Passo:** Acompanhe cada operação elementar de linha (escalonamento) explicada detalhadamente.
* **Visualização Geométrica:**
    *  **2D:** Visualização de retas e interseções para sistemas 2x2.
    *  **3D:** Visualização de planos no espaço para sistemas 3x3.
* **Suporte a Múltiplas Dimensões:** Resolve sistemas de 2 até 5 variáveis.
* **Relatórios Profissionais:** Exportação da resolução completa em formato HTML (pronto para impressão em PDF) com formatação LaTeX de alta qualidade.
* **Interface Responsiva:** Design limpo e moderno construído com Streamlit.

## 🛠️ Tecnologias Utilizadas

Este projeto foi construído seguindo uma arquitetura modular moderna:

* **[Python 3.10+](https://www.python.org/)**: Linguagem base.
* **[Streamlit](https://streamlit.io/)**: Framework para a interface web interativa.
* **[NumPy](https://numpy.org/)**: Motor de cálculo numérico e álgebra linear.
* **[Plotly](https://plotly.com/)**: Biblioteca para geração de gráficos interativos 2D e 3D.
* **MathJax/LaTeX**: Para renderização matemática precisa das equações.

## 🚀 Como Executar Localmente

Siga os passos abaixo para rodar o projeto na sua máquina:

1.  **Clone o repositório:**
    ```bash
    git clone [https://github.com/yzkjr/streamliSL.git](https://github.com/yzkjr/streamliSL.git)
    cd streamliSL
    ```

2.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Execute a aplicação:**
    ```bash
    streamlit run app.py
    ```

4.  O projeto abrirá automaticamente no seu navegador em `http://localhost:8501`.

## 📂 Estrutura do Projeto

O código foi refatorado para garantir manutenibilidade e escalabilidade:

```text
├── app.py                # Orquestrador principal da aplicação
├── requirements.txt      # Lista de dependências
└── modules/              # Módulos do sistema
    ├── logic.py          # Lógica matemática (Gauss-Jordan)
    ├── graphics.py       # Geração de gráficos Plotly
    ├── interface.py      # Componentes de UI e animações
    └── ui.py             # Configurações de layout e CSS
