import streamlit as st
import chess
import chess.pgn
import chess.svg
import streamlit.components.v1 as components
import io
import os
from modules import (
    load_data,
    NeuralChessEvaluator,
    EnsembleModel,
    load_model,
)

# Configuración inicial de la página
def set_page_config():
    """Configura la página de Streamlit con título, ícono y diseño."""
    st.set_page_config(
        page_title="Análisis Básico de Ajedrez",
        page_icon="♟️",
        layout="centered",
        initial_sidebar_state="expanded",
    )
    # CSS personalizado para ajustar el diseño de la aplicación
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f5f5f5;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Inicialización del estado de sesión
def initialize_session_state():
    """Inicializa las variables de estado de sesión para la aplicación."""
    if "evaluator" not in st.session_state:
        st.session_state.evaluator = NeuralChessEvaluator(input_shape=18)
    if "ensemble_model" not in st.session_state:
        st.session_state.ensemble_model = EnsembleModel()
    if "model_loaded" not in st.session_state:
        st.session_state.model_loaded = False
    if "threshold" not in st.session_state:
        st.session_state.threshold = 0.3
    if "current_move_index" not in st.session_state:
        st.session_state.current_move_index = -1
    if "moves" not in st.session_state:
        st.session_state.moves = []
    if "board" not in st.session_state:
        st.session_state.board = chess.Board()

# Gestión de modelos en la barra lateral
def model_management_sidebar():
    """Gestión de modelos desde la barra lateral."""
    st.sidebar.header("🧠 Gestión de Modelos")
    # Crear directorio de modelos si no existe
    if not os.path.exists("./modelos"):
        os.makedirs("./modelos")

    # Listar modelos disponibles
    model_files = [f for f in os.listdir("./modelos") if f.endswith("_ensemble.joblib")]
    if model_files:
        selected_model = st.sidebar.selectbox("Modelos disponibles", model_files)
        model_base = selected_model.replace("_ensemble.joblib", "")
        if st.sidebar.button("Cargar Modelo"):
            try:
                model_path = f"./modelos/{model_base}"
                # Verificar archivos del modelo
                model_dir_files = [
                    f for f in os.listdir("./modelos") if model_base in f
                ]

                # Verificar integridad del modelo
                required_patterns = [
                    "_evaluator",
                    "_ensemble.joblib",
                    "_metadata.joblib",
                ]
                missing_files = []
                for pattern in required_patterns:
                    found = False
                    for file in model_dir_files:
                        if pattern in file:
                            found = True
                            break
                    if not found:
                        missing_files.append(pattern)

                if missing_files:
                    st.sidebar.error(f"Faltan archivos: {missing_files}")
                    raise ValueError(
                        f"Archivos del modelo incompletos: faltan {missing_files}"
                    )

                # Intentar cargar el modelo
                st.session_state.evaluator, st.session_state.ensemble_model, params = (
                    load_model(model_path)
                )
                st.session_state.model_loaded = True
                st.sidebar.success("Modelo cargado correctamente")
            except Exception as e:
                st.sidebar.error(f"Error al cargar: {str(e)}")
    else:
        st.sidebar.warning("No hay modelos guardados disponibles")
        st.sidebar.info(
            "Por favor, entrene un modelo primero usando la interfaz avanzada."
        )

# Actualización de la visualización del tablero
def update_board_display():
    """Actualiza la visualización del tablero de ajedrez."""
    board = chess.Board()
    for i in range(st.session_state.current_move_index + 1):
        if i < len(st.session_state.moves):
            board.push(st.session_state.moves[i])

    # Guardar el tablero actual en el estado de sesión
    st.session_state.board = board

    # Generar SVG con resaltado del último movimiento
    lastmove = None
    if (
        st.session_state.current_move_index >= 0
        and st.session_state.current_move_index < len(st.session_state.moves)
    ):
        lastmove = st.session_state.moves[st.session_state.current_move_index]

    board_svg = chess.svg.board(
        board=board,
        size=600,
        lastmove=lastmove,
        check=board.king(board.turn) if board.is_check() else None,
    )

    return board_svg

# Sección de análisis de partidas
def game_analysis_section():
    """Sección principal para cargar y analizar partidas de ajedrez."""
    st.header("📊 Análisis Básico de Partidas")

    # Cargar archivo de partida
    uploaded_file = st.file_uploader(
        "Cargar partida a analizar (PGN/TXT)",
        type=["pgn", "txt"],
        help="Sube un archivo PGN estándar o un TXT con notación algebraica",
    )

    if uploaded_file and st.session_state.model_loaded:
        if st.button("⚡ Iniciar Análisis"):
            with st.spinner("🔍 Procesando partida..."):
                try:
                    # Determinar el tipo de archivo
                    file_extension = uploaded_file.name.split(".")[-1].lower()

                    if file_extension == "pgn":
                        # Procesar archivo PGN
                        uploaded_file.seek(
                            0
                        )  # Asegurar que estamos al inicio del archivo
                        pgn_content = uploaded_file.read().decode("utf-8")
                        game = chess.pgn.read_game(io.StringIO(pgn_content))

                        if game is None:
                            raise ValueError(
                                "No se pudo leer la partida del archivo PGN"
                            )

                        # Extraer movimientos
                        board = game.board()
                        node = game
                        st.session_state.moves = []

                        while node.variations:
                            next_node = node.variation(0)
                            st.session_state.moves.append(next_node.move)
                            node = next_node

                        # Reiniciar el índice de movimiento
                        st.session_state.current_move_index = -1
                        st.session_state.board = chess.Board()

                        # Crear tablero final para predicción
                        final_board = chess.Board()
                        for move in st.session_state.moves:
                            final_board.push(move)

                    else:
                        # Procesar TXT u otros formatos
                        df = load_data(uploaded_file)

                        if df is None or df.empty:
                            raise ValueError(
                                "Error al cargar el archivo o archivo vacío"
                            )

                        if "Moves" not in df.columns:
                            raise ValueError(
                                "Formato inválido: columna 'Moves' no encontrada"
                            )

                        # Obtener movimientos
                        last_game = df.iloc[0]
                        moves_str = last_game["Moves"]

                        # Asegurar que moves_str sea un string
                        if isinstance(moves_str, list):
                            moves_str = " ".join(moves_str)

                        # Reiniciar el tablero y procesar movimientos
                        board = chess.Board()
                        st.session_state.moves = []

                        for move_str in moves_str.split():
                            try:
                                move = chess.Move.from_uci(move_str)
                                if board.is_legal(move):
                                    st.session_state.moves.append(move)
                                    board.push(move)
                                else:
                                    st.warning(
                                        f"Movimiento ilegal ignorado: {move_str}"
                                    )
                            except Exception as e:
                                st.warning(
                                    f"Error al procesar movimiento {move_str}: {str(e)}"
                                )

                        # Reiniciar el índice de movimiento
                        st.session_state.current_move_index = -1
                        st.session_state.board = chess.Board()
                        final_board = board

                    # Predicción final con verificación
                    try:
                        final_prediction = st.session_state.ensemble_model.predict(
                            final_board
                        )
                        if not isinstance(final_prediction, dict):
                            raise TypeError("Formato de predicción inválido")
                    except Exception as e:
                        raise ValueError(f"Error en predicción final: {str(e)}")

                    # Mostrar resultados
                    display_results(final_prediction)

                except Exception as e:
                    st.error(f"❌ Error inesperado: {str(e)}")

    elif not st.session_state.model_loaded:
        st.warning("⚠️ Por favor, cargue un modelo primero desde el panel lateral.")
    elif not uploaded_file:
        st.info("ℹ️ Suba un archivo PGN o TXT para analizar.")

    st.subheader("Resultados del Análisis")

    # Visualización del tablero
    st.header("Tablero Actual")

    # Crear un placeholder para el tablero
    board_placeholder = st.empty()

    # Mostrar el tablero inicial
    current_svg = update_board_display()
    board_placeholder.image(current_svg, width=600)

    # Botones para navegar por la partida
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("⏮️"):  # Ir al inicio
            st.session_state.current_move_index = -1
            current_svg = update_board_display()
            board_placeholder.image(current_svg, width=600)

    with col2:
        if st.button("⏪"):  # Retroceder un movimiento
            if st.session_state.current_move_index >= 0:
                st.session_state.current_move_index -= 1
                current_svg = update_board_display()
                board_placeholder.image(current_svg, width=600)

    with col3:
        if st.button("⏩"):  # Avanzar un movimiento
            if st.session_state.current_move_index < len(st.session_state.moves) - 1:
                st.session_state.current_move_index += 1
                current_svg = update_board_display()
                board_placeholder.image(current_svg, width=600)

    with col4:
        if st.button("⏭️"):  # Ir al final
            st.session_state.current_move_index = len(st.session_state.moves) - 1
            current_svg = update_board_display()
            board_placeholder.image(current_svg, width=600)

    # Mostrar FEN de la posición actual
    st.subheader("Posición FEN")
    st.code(st.session_state.board.fen())

    # Mostrar historial de movimientos
    st.subheader("Historial de Movimientos")
    if st.session_state.moves:
        move_history = ""
        board = chess.Board()
        for i, move in enumerate(st.session_state.moves):
            if i % 2 == 0:
                move_number = i // 2 + 1
                move_history += f"{move_number}. {board.san(move)} "
            else:
                move_history += f"{board.san(move)} "
            board.push(move)
        st.write(move_history)
    else:
        st.write("No hay movimientos registrados.")

# Mostrar resultados del análisis
def display_results(prediction):
    """Muestra los resultados del análisis de la partida."""
    st.subheader("Resultados del Análisis")

    # Mostrar predicción
    pred_value = prediction.get("prediccion", 0)

    if pred_value > 1.5:
        result_text = "Victoria blancas"
    elif pred_value < 0.5:
        result_text = "Victoria negras"
    else:
        result_text = "Tablas"

    with st.container():
        st.markdown(
            f"""
            <div style="padding: 20px; border-radius: 10px; background-color: #e6f3ff; margin-bottom: 20px;">
                <h3 style="text-align: center; margin-bottom: 10px;">Predicción del Resultado</h3>
                <h2 style="text-align: center; color: #0066cc;">{result_text}</h2>
                <p style="text-align: center;">Confianza: {prediction["confianza"]:.2%}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# Ejecución principal
def main():
    """Función principal para ejecutar la aplicación."""
    set_page_config()
    initialize_session_state()

    # Encabezado principal con logo
    col1, col2 = st.columns([1, 5])
    with col1:
        st.image("https://cdn-icons-png.flaticon.com/512/4363/4363854.png", width=80)
    with col2:
        st.title("Análisis Básico de Ajedrez")
        st.markdown("*Visualización de partidas y predicción de resultados*")

    # Sidebar para gestión de modelos
    model_management_sidebar()

    # Opción para acceder a la interfaz avanzada
    if st.sidebar.button("🚀 Acceder a Interfaz Avanzada"):
        import subprocess

        try:
            st.sidebar.success("Iniciando interfaz avanzada...")
            subprocess.Popen(["streamlit", "run", "ajedrez_pro.py"])
            st.sidebar.info(
                "Accediendo a  la interfaz Avanzada. Por favor, espera..."
            )
        except Exception as e:
            st.sidebar.error(f"Error al iniciar la interfaz avanzada: {str(e)}")

    # Sección principal de análisis
    game_analysis_section()

if __name__ == "__main__":
    main()
