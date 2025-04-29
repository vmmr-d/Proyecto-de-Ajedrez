# -*- coding: utf-8 -*-
# !/usr/bin/env python
# =============================================================================
# === Imports =================================================================
# =============================================================================
import math
import streamlit as st  # Import Streamlit first
import streamlit.components.v1 as components
import chess
import chess.svg
import chess.pgn
import base64
import pandas as pd
import os
import io
import numpy as np
from datetime import datetime
import plotly.graph_objects as go
import logging  # Added for logging errors

try:
    from modules import (
        load_data,
        NeuralChessEvaluator,
        ChessAnalyzer,
        prepare_training_data,
        EnsembleModel,
        generate_report,
        generate_pdf_report,
        visualize_position,
        save_model,
        load_model,
    )
except ImportError as e:
    st.error(
        f"Error importing 'modules.py': {e}. Make sure the file exists and has the necessary components."
    )
    st.stop()  # Stop execution if modules cannot be imported
except NameError as ne:
    # Catch if specific classes/functions are missing within modules.py
    st.error(
        f"Error: A required component is missing from 'modules.py': {ne}. Please check the file content."
    )
    st.stop()

# =============================================================================
# === Streamlit Page Configuration (MUST BE FIRST st command) ================
# =============================================================================
st.set_page_config(
    page_title="Análisis Avanzado de Ajedrez",
    page_icon="♟️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# === Logging Setup ===========================================================
# =============================================================================
# Configure logging to show info level messages
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


# =============================================================================
# === CSS y Personalización Visual ============================================
# =============================================================================
def set_transparent_background():
    if os.path.exists("chess_bg.png"):
        bg_path = os.path.join(os.getcwd(), "chess_bg.png")
        with open(bg_path, "rb") as f:
            bg_base64 = base64.b64encode(f.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background-image: url("data:image/png;base64,{bg_base64}");
                background-size: cover;
                background-repeat: no-repeat;
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def apply_custom_styles():
    """Aplica estilos CSS personalizados a la aplicación."""
    css = """
    <style>
        .main-header {
            font-size: 2.5rem;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 1rem;
        }
        .sub-header {
            font-size: 1.8rem;
            color: #2563EB;
            margin-top: 1rem;
            margin-bottom: 0.5rem;
        }
        .board-container {
            display: flex;
            justify-content: center;
            margin: 1rem 0;
        }
        .info-box {
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 5px;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .nav-button {
            font-weight: bold;
            width: 100%;
        }
        .critical-move {
            background-color: rgba(239, 68, 68, 0.2);
            border-left: 4px solid #EF4444;
            padding: 0.5rem;
            margin: 0.5rem 0;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# =============================================================================
# === Funciones para Manejo de Errores ========================================
# =============================================================================
def log_error(context, error):
    """Registra errores en el log y en session_state para referencia."""
    error_msg = f"{context}: {str(error)}"
    logging.error(error_msg)
    if "error_log" not in st.session_state:
        st.session_state.error_log = []
    st.session_state.error_log.append(error_msg)


# =============================================================================
# === Session State Initialization ============================================
# =============================================================================
def initialize_session_state():
    """Inicializa las variables de estado de la sesión si no existen."""
    # Inicialización de evaluador y modelo ensemble
    if "evaluator" not in st.session_state:
        try:
            st.session_state.evaluator = NeuralChessEvaluator(input_shape=18)
            logging.info("NeuralChessEvaluator inicializado.")
        except Exception as e:
            st.error(f"Error inicializando NeuralChessEvaluator: {e}")
            logging.error(f"Error inicializando NeuralChessEvaluator: {e}")
            st.session_state.evaluator = None

    if "ensemble_model" not in st.session_state:
        try:
            st.session_state.ensemble_model = EnsembleModel()
            logging.info("EnsembleModel inicializado.")
            # Asignar evaluador al ensemble model si ambos están inicializados
            if st.session_state.evaluator and st.session_state.ensemble_model:
                st.session_state.ensemble_model.set_evaluator(
                    st.session_state.evaluator
                )
                logging.info("Evaluador asignado al EnsembleModel.")
        except Exception as e:
            st.error(f"Error inicializando EnsembleModel: {e}")
            logging.error(f"Error inicializando EnsembleModel: {e}")
            st.session_state.ensemble_model = None

    # Otras variables de estado
    default_states = {
        "model_loaded": False,
        "threshold": 0.3,
        "current_move_index": -1,  # -1 indica posición inicial
        "moves": [],
        "board": chess.Board(),
        "analysis_mode": "básico",
        "error_log": [],
        "model_metadata": None,
        "critical_moves": None,
        "evaluations": [],
        "move_numbers": [],
        "board_history": [],
        "game_phase_filter": ["Apertura", "Medio juego", "Final"],
    }

    for key, value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = value


# =============================================================================
# === Renderizado del Tablero =================================================
# =============================================================================
def render_chess_board(board, size=400, lastmove=None, check_square=None, arrows=None):
    """Renderiza el tablero usando SVG incrustado en Markdown."""
    try:
        svg = chess.svg.board(
            board=board,
            size=size,
            lastmove=lastmove,
            check=check_square,
            arrows=arrows if arrows else [],
        )
        b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
        html = f'<div class="board-container"><img src="data:image/svg+xml;base64,{b64}" /></div>'
        st.markdown(html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error renderizando tablero: {e}")
        log_error("render_chess_board", e)


def interactive_board(board_history, critical_moves=None):
    """Tablero interactivo con controles de navegación y resaltado de movimientos críticos."""
    if not board_history:
        st.warning("No hay posiciones para mostrar en el tablero")
        board = chess.Board()
        render_chess_board(board, size=400)
        return

    # Controles de navegación
    col1, col2, col3, col4 = st.columns([1, 1, 1, 1])

    with col1:
        if st.button("⏮️ Inicio", disabled=st.session_state.current_move_index <= 0):
            st.session_state.current_move_index = 0
            st.rerun()

    with col2:
        if st.button("⬅️ Anterior", disabled=st.session_state.current_move_index <= 0):
            st.session_state.current_move_index -= 1
            st.rerun()

    with col3:
        if st.button(
            "➡️ Siguiente",
            disabled=st.session_state.current_move_index >= len(board_history) - 1,
        ):
            st.session_state.current_move_index += 1
            st.rerun()

    with col4:
        if st.button(
            "⏭️ Final",
            disabled=st.session_state.current_move_index >= len(board_history) - 1,
        ):
            st.session_state.current_move_index = len(board_history) - 1
            st.rerun()

    # Slider interactivo
    max_value = len(board_history) - 1
    if max_value > 0:
        target_index = st.slider(
            "Movimiento", 0, max_value, st.session_state.current_move_index
        )
        if target_index != st.session_state.current_move_index:
            st.session_state.current_move_index = target_index
            st.rerun()

    # Obtener el tablero actual y el último movimiento
    current_board = board_history[st.session_state.current_move_index]

    # Determinar el último movimiento para resaltarlo
    lastmove = None
    arrows = []
    if st.session_state.current_move_index > 0 and st.session_state.moves:
        move_idx = st.session_state.current_move_index - 1
        if move_idx < len(st.session_state.moves):
            lastmove = st.session_state.moves[move_idx]
            if hasattr(lastmove, "from_square") and hasattr(lastmove, "to_square"):
                arrows.append((lastmove.from_square, lastmove.to_square))

    # Verificar si es un movimiento crítico
    is_critical = False
    critical_info = None
    if critical_moves is not None and not critical_moves.empty:
        # Buscar si el movimiento actual está en la lista de críticos
        for _, move in critical_moves.iterrows():
            if move["move_number"] == (st.session_state.current_move_index + 1) // 2:
                is_critical = True
                critical_info = move
                break

    # Renderizar el tablero
    check_square = (
        current_board.king(current_board.turn) if current_board.is_check() else None
    )
    render_chess_board(
        current_board,
        size=500,
        lastmove=lastmove,
        check_square=check_square,
        arrows=arrows,
    )

    # Mostrar información contextual
    st.markdown(
        f"**Movimiento actual: {st.session_state.current_move_index}/{max_value}**"
    )

    if is_critical and critical_info is not None:
        st.markdown(
            f"""
            <div class="critical-move">
                <strong>¡Movimiento crítico!</strong><br>
                Jugador: {critical_info["player"]}<br>
                Diferencia de evaluación: {critical_info["eval_diff"]:.3f}<br>
                Fase: {critical_info["game_phase"]}
            </div>
            """,
            unsafe_allow_html=True,
        )


# =============================================================================
# === Gestión de Modelos (Sidebar) ============================================
# =============================================================================
def model_management_sidebar():
    """Gestiona la carga y entrenamiento de modelos desde la barra lateral."""
    st.sidebar.header("🧠 Gestión de Modelos")

    model_dir = "./modelos"
    if not os.path.exists(model_dir):
        try:
            os.makedirs(model_dir)
            logging.info(f"Directorio de modelos creado: {model_dir}")
        except OSError as e:
            st.sidebar.error(f"No se pudo crear el directorio de modelos: {e}")
            log_error("model_management_sidebar_create_dir", e)
            return

    model_option = st.sidebar.radio(
        "Modo de modelo", ["Pre-entrenado", "Entrenar nuevo"], key="model_mode_radio"
    )

    # Selector de umbral para movimientos críticos
    st.session_state.threshold = st.sidebar.slider(
        "Umbral para movimientos críticos",
        min_value=0.1,
        max_value=1.0,
        value=st.session_state.threshold,
        step=0.05,
        help="Diferencia mínima en la evaluación para considerar un movimiento como crítico",
    )

    # Filtro de fases de juego (del segundo archivo)
    st.session_state.game_phase_filter = st.sidebar.multiselect(
        "Filtrar por fase de juego",
        options=["Apertura", "Medio juego", "Final"],
        default=st.session_state.game_phase_filter,
    )

    if model_option == "Pre-entrenado":
        load_existing_model(model_dir)
    else:
        train_new_model(model_dir)


def load_existing_model(model_dir):
    """Carga un modelo pre-entrenado, verificando archivos relacionados."""
    try:
        # Listar todos los archivos en el directorio de modelos
        all_files_in_modelos_dir = os.listdir(model_dir)

        # Identificar modelos buscando archivos de metadatos
        metadata_files = [
            f for f in all_files_in_modelos_dir if f.endswith("_metadata.joblib")
        ]

        # Extraer los nombres base (prefijos) de los modelos
        model_names = sorted(
            [f.replace("_metadata.joblib", "") for f in metadata_files],
            reverse=True,  # Ordenar por nombre (fecha)
        )

        logging.info(f"Modelos encontrados: {model_names}")
    except FileNotFoundError:
        st.sidebar.warning(f"Directorio de modelos no encontrado: '{model_dir}'")
        logging.warning(f"Directorio de modelos no encontrado: '{model_dir}'")
        return
    except Exception as e:
        st.sidebar.error(f"Error listando modelos: {e}")
        log_error("load_existing_model_list", e)
        return

    if model_names:
        selected_model_name = st.sidebar.selectbox(
            "Modelos disponibles",
            model_names,
            key="model_select",
            index=0 if model_names else None,
        )

        if st.sidebar.button("Cargar Modelo", key="load_model_button"):
            if not selected_model_name:
                st.sidebar.warning("Por favor, seleccione un modelo.")
                return

            # Verificación de archivos del modelo
            model_path_prefix = os.path.join(model_dir, selected_model_name)
            logging.info(f"Intentando cargar modelo con prefijo: {model_path_prefix}")

            # Definir los sufijos esperados para un modelo completo
            required_suffixes = [
                "_evaluator_model.h5",
                "_evaluator_scaler.joblib",
                "_ensemble.joblib",
                "_metadata.joblib",
            ]

            missing_files_info = []

            # Verificar si existe cada archivo requerido usando el prefijo
            logging.info(
                f"Verificando archivos requeridos con prefijo '{selected_model_name}' en '{model_dir}'"
            )

            for suffix in required_suffixes:
                expected_filename = selected_model_name + suffix
                expected_filepath = os.path.join(model_dir, expected_filename)
                if not os.path.exists(expected_filepath):
                    missing_files_info.append(expected_filename)
                    logging.warning(
                        f"Archivo requerido no encontrado: {expected_filepath}"
                    )

            if missing_files_info:
                st.sidebar.error(
                    f"Faltan archivos para '{selected_model_name}': {', '.join(missing_files_info)}"
                )
                log_error(
                    "load_existing_model_verify",
                    f"Archivos faltantes: {missing_files_info} para prefijo {model_path_prefix}",
                )
                return

            # Cargar el modelo usando la ruta del prefijo
            try:
                with st.spinner(f"Cargando modelo '{selected_model_name}'..."):
                    # load_model debe devolver evaluator, ensemble, metadata
                    evaluator, ensemble, metadata = load_model(model_path_prefix)

                    # Validación básica de objetos cargados
                    if not isinstance(
                        evaluator, NeuralChessEvaluator
                    ) or not isinstance(ensemble, EnsembleModel):
                        raise TypeError(
                            "La función load_model no devolvió los tipos de objeto esperados (Evaluator, Ensemble)."
                        )

                    st.session_state.evaluator = evaluator
                    st.session_state.ensemble_model = ensemble
                    st.session_state.model_loaded = True
                    st.session_state.model_metadata = metadata  # Almacenar metadatos

                    logging.info(
                        f"Modelo '{selected_model_name}' cargado exitosamente."
                    )
                    st.sidebar.success(f"Modelo '{selected_model_name}' cargado.")

                    # Asignar evaluador al modelo ensemble explícitamente después de cargar
                    if st.session_state.evaluator and st.session_state.ensemble_model:
                        st.session_state.ensemble_model.set_evaluator(
                            st.session_state.evaluator
                        )
                        logging.info(
                            "Evaluador reasignado al EnsembleModel después de cargar."
                        )

                    # Mostrar detalles de metadatos
                    if metadata and isinstance(metadata, dict):
                        st.sidebar.markdown("**Detalles del Modelo Cargado:**")
                        params_to_show = metadata.get(
                            "params", metadata.get("training_params", {})
                        )  # Verificar ambas claves posibles

                        st.sidebar.caption(
                            f"Entrenado: {metadata.get('timestamp', 'N/A')}"
                        )
                        st.sidebar.caption(
                            f"Archivo PGN: {metadata.get('trained_on', 'N/A')}"
                        )
                        st.sidebar.caption(
                            f"Épocas: {params_to_show.get('epochs', 'N/A')}"
                        )
                        st.sidebar.caption(
                            f"Batch Size: {params_to_show.get('batch_size', 'N/A')}"
                        )
                        st.sidebar.caption(f"LR: {params_to_show.get('lr', 'N/A')}")
                        st.sidebar.caption(
                            f"Max Juegos: {params_to_show.get('max_games', 'N/A')}"
                        )
                    else:
                        st.sidebar.info(
                            "No se encontraron metadatos detallados para este modelo."
                        )
            except FileNotFoundError as fnf_error:
                st.sidebar.error(
                    f"Error al cargar: No se encontró un archivo esperado durante la carga. Verifique que todos los componentes del modelo ({selected_model_name}*) existan. Detalles: {fnf_error}"
                )
                log_error("load_existing_model_load_fnf", fnf_error)
                st.session_state.model_loaded = False
            except TypeError as te:
                st.sidebar.error(
                    f"Error de tipo al cargar: {te}. Podría indicar un problema con los archivos guardados o la función load_model."
                )
                log_error("load_existing_model_load_type", te)
                st.session_state.model_loaded = False
            except Exception as e:
                st.sidebar.error(f"Error inesperado al cargar el modelo: {str(e)}")
                log_error("load_existing_model_load", e)
                st.session_state.model_loaded = (
                    False  # Restablecer estado en caso de error
                )
    else:
        st.sidebar.info(
            "No hay modelos guardados disponibles en la carpeta 'modelos'. Busque archivos `*_metadata.joblib`."
        )


@st.cache_data  # Cachear el paso de preparación si las entradas son las mismas
def cached_prepare_training_data(uploaded_file_content, max_games):
    """Wrapper cacheado para prepare_training_data"""
    logging.info(
        f"Preparando datos de entrenamiento (max_games={max_games}). Cache {'hit' if '_cache_hits' in st.session_state else 'miss'}."
    )
    # Pasamos el contenido en lugar del objeto de archivo para el caché
    file_like_object = io.BytesIO(uploaded_file_content)

    # Asegurar que prepare_training_data sea robusto
    try:
        return prepare_training_data(file_like_object, max_games)
    except Exception as e:
        log_error("cached_prepare_training_data", e)
        st.error(f"Error crítico durante la preparación de datos: {e}")
        return None  # Devolver None en caso de error crítico


def train_new_model(model_dir):
    """Entrena un nuevo modelo basado en datos subidos."""
    training_file = st.sidebar.file_uploader(
        "Subir datos de entrenamiento (PGN)", type=["pgn"], key="train_uploader"
    )

    # Parámetros de entrenamiento
    if training_file:
        with st.sidebar.expander("⚙️ Parámetros de Entrenamiento"):
            epochs = st.number_input(
                "Épocas",
                min_value=1,
                max_value=1000,
                value=10,
                step=1,
                key="train_epochs",
            )

            batch_size = st.select_slider(
                "Tamaño de lote",
                options=[32, 64, 128, 256, 512],
                value=256,
                key="train_batch",
            )

            learning_rate = st.number_input(
                "Tasa de aprendizaje",
                min_value=1e-6,
                max_value=1e-1,
                value=1e-3,
                step=1e-4,
                format="%.5f",
                key="train_lr",
            )

            max_games_to_process = st.number_input(
                "Máx. partidas a procesar",
                min_value=100,
                value=5000,
                step=100,
                key="train_max_games",
            )

        if st.sidebar.button("🚀 Entrenar Modelo", key="train_button"):
            # Validar que los modelos base estén inicializados
            if (
                st.session_state.evaluator is None
                or st.session_state.ensemble_model is None
            ):
                st.sidebar.error(
                    "Modelos base (Evaluator/Ensemble) no inicializados. Verifique 'modules.py' y reinicie."
                )
                logging.error(
                    "Modelos base no inicializados durante intento de entrenamiento."
                )
                return

            st.sidebar.info("Iniciando proceso de entrenamiento...")
            progress_bar = st.sidebar.progress(0, text="Preparando datos...")

            try:
                # 1. Preparar Datos
                uploaded_file_content = training_file.getvalue()
                games_df = cached_prepare_training_data(
                    uploaded_file_content, max_games_to_process
                )
                progress_bar.progress(25, text="Datos preparados.")

                if games_df is None or games_df.empty or "fen" not in games_df.columns:
                    raise ValueError(
                        "La preparación de datos falló o no devolvió un DataFrame válido con FEN."
                    )

                # 2. Extraer Tableros
                progress_bar.progress(40, text="Extrayendo posiciones (FEN)...")
                boards = []
                invalid_fen_count = 0

                for fen in games_df["fen"]:
                    try:
                        boards.append(chess.Board(fen))
                    except ValueError:
                        invalid_fen_count += 1
                        continue  # Omitir FENs inválidos

                if invalid_fen_count > 0:
                    logging.warning(
                        f"{invalid_fen_count} FEN inválidos omitidos durante el entrenamiento."
                    )

                if not boards:
                    raise ValueError(
                        "No se pudieron generar tableros válidos desde los datos FEN."
                    )

                logging.info(f"Extraídas {len(boards)} posiciones de tablero válidas.")
                progress_bar.progress(50, text="Posiciones extraídas.")

                # 3. Entrenar Evaluador
                progress_bar.progress(60, text="Entrenando Evaluador Neural...")

                # Asegurar que la instancia del evaluador sea válida
                if not isinstance(st.session_state.evaluator, NeuralChessEvaluator):
                    raise TypeError(
                        "st.session_state.evaluator no es una instancia válida."
                    )

                training_params_eval = {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                }

                st.session_state.evaluator.train_batch(
                    games=boards,  # Pasar objetos de tablero
                    **training_params_eval,
                )

                logging.info("Evaluador neural entrenado.")
                progress_bar.progress(80, text="Evaluador entrenado.")

                # 4. Guardar Modelo
                progress_bar.progress(90, text="Guardando modelo...")
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                model_name = f"chess_model_{timestamp}"  # Nombre estandarizado
                model_path_prefix = os.path.join(model_dir, model_name)

                metadata_to_save = {
                    "trained_on": training_file.name,
                    "timestamp": timestamp,
                    "params": {  # Clave consistente para parámetros
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "lr": learning_rate,
                        "max_games": max_games_to_process,
                    },
                    "evaluator_input_shape": st.session_state.evaluator.input_shape,
                }

                # Llamar a save_model desde model_trainer.py
                save_model(
                    st.session_state.evaluator,
                    st.session_state.ensemble_model,
                    model_path_prefix,  # Pasar el prefijo
                    metadata_to_save,  # Pasar el diccionario de metadatos
                )

                # Añadir logging explícito después del intento de guardado
                logging.info(
                    f"Intento de guardar componentes del modelo con prefijo: {model_path_prefix}"
                )

                # Listar archivos creados para verificación
                try:
                    saved_files = [
                        f for f in os.listdir(model_dir) if f.startswith(model_name)
                    ]
                    logging.info(
                        f"Archivos encontrados después de guardar: {saved_files}"
                    )

                    if not saved_files:
                        st.sidebar.warning(
                            "No se encontraron archivos guardados, la operación podría haber fallado silenciosamente."
                        )
                except Exception as list_e:
                    logging.error(
                        f"No se pudieron listar archivos después de guardar: {list_e}"
                    )

                st.session_state.model_loaded = (
                    True  # Marcar como cargado ya que está listo
                )
                st.session_state.model_metadata = (
                    metadata_to_save  # Almacenar metadatos del modelo recién entrenado
                )

                progress_bar.progress(100, text="¡Modelo entrenado y guardado!")
                st.sidebar.success(f"Modelo entrenado y guardado como '{model_name}'.")

                # Actualizar la lista de modelos en la barra lateral si es posible
                st.rerun()  # Rerun para actualizar la lista de modelos

            except MemoryError:
                progress_bar.empty()
                st.sidebar.error(
                    "Error de memoria durante el entrenamiento. Intente con menos partidas o un tamaño de lote más pequeño."
                )
                log_error("train_new_model_memory", "MemoryError")
            except ValueError as ve:
                progress_bar.empty()
                st.sidebar.error(f"Error en los datos o parámetros: {ve}")
                log_error("train_new_model_value_error", ve)
            except TypeError as te:
                progress_bar.empty()
                st.sidebar.error(
                    f"Error de tipo: {te}. Verifique las clases del modelo."
                )
                log_error("train_new_model_type_error", te)
            except Exception as e:
                progress_bar.empty()
                st.sidebar.error(f"Error inesperado en entrenamiento: {str(e)}")
                log_error("train_new_model", e)


# =============================================================================
# === Sistema de Análisis de Partidas =========================================
# =============================================================================
def game_analysis_system():
    """Maneja la carga de archivos y activa el análisis de partidas."""
    st.header("🎮 Análisis de Partida")

    uploaded_file = st.file_uploader(
        "Subir partida (PGN o TXT con columna 'Moves')",
        type=["pgn", "txt"],
        key="game_uploader",
    )

    # Verificar si un modelo está listo
    model_ready = (
        st.session_state.model_loaded and st.session_state.evaluator is not None
    )

    if not model_ready:
        st.warning(
            "⚠️ Por favor, cargue o entrene un modelo desde la barra lateral para iniciar el análisis."
        )
        return  # Retornar temprano si no hay modelo

    if uploaded_file:
        # Botón para iniciar explícitamente el análisis después de la carga
        if st.button("🔍 Analizar Partida", key="analyze_button", type="primary"):
            st.session_state.current_move_index = (
                -1
            )  # Resetear índice para nueva partida
            st.session_state.moves = []  # Limpiar movimientos anteriores
            st.session_state.board = chess.Board()  # Resetear tablero
            st.session_state.error_log = []  # Limpiar errores anteriores
            st.session_state.board_history = []  # Limpiar historial de tableros
            st.session_state.evaluations = []  # Limpiar evaluaciones anteriores
            st.session_state.move_numbers = []  # Limpiar números de movimiento
            st.session_state.critical_moves = None  # Limpiar movimientos críticos

            process_game_file(uploaded_file)
    else:
        # Limpiar datos de partida anterior si no hay archivo cargado
        if st.session_state.moves:
            logging.info("No hay archivo cargado, limpiando datos de partida anterior.")
            st.session_state.moves = []
            st.session_state.current_move_index = -1
            st.session_state.board = chess.Board()
            st.session_state.error_log = []
            st.session_state.board_history = []
            st.session_state.evaluations = []
            st.session_state.move_numbers = []
            st.session_state.critical_moves = None


def process_game_file(uploaded_file):
    """Procesa el archivo de partida cargado (PGN o TXT)."""
    try:
        file_name = uploaded_file.name
        file_extension = os.path.splitext(file_name)[1].lower()
        logging.info(f"Procesando archivo: {file_name} (Extensión: {file_extension})")

        with st.spinner(f"Procesando archivo '{file_name}'..."):
            if file_extension == ".pgn":
                process_pgn_file(uploaded_file)
            elif file_extension == ".txt":
                process_txt_file(uploaded_file)
            else:
                st.error(f"Formato de archivo no soportado: {file_extension}")
                return  # Detener procesamiento

            # Analizar la partida si se cargaron movimientos
            if st.session_state.moves:
                analyze_loaded_game()

                # Si el procesamiento fue exitoso y se cargaron movimientos
                st.success(
                    f"Archivo '{file_name}' procesado. {len(st.session_state.moves)} movimientos cargados. Navegue por la partida."
                )
            else:
                # Este caso podría ocurrir si el archivo tenía formato válido pero no contenía movimientos válidos
                st.warning(
                    f"Archivo '{file_name}' procesado, pero no se encontraron movimientos válidos para analizar."
                )
    except ValueError as ve:
        # Capturar errores específicos de validación de las funciones de procesamiento
        st.error(f"Error de formato o contenido en el archivo: {ve}")
        log_error("process_game_file_value_error", ve)
        # Limpiar datos potencialmente cargados parcialmente en caso de error
        st.session_state.moves = []
        st.session_state.current_move_index = -1
    except Exception as e:
        st.error(f"Error inesperado procesando el archivo: {str(e)}")
        log_error("process_game_file_generic", e)
        st.session_state.moves = []
        st.session_state.current_move_index = -1


# Usar caché para el análisis de PGN si el contenido no cambia
@st.cache_data
def parse_pgn_content(pgn_content_str):
    """Analiza el contenido PGN y devuelve la lista de movimientos de la línea principal."""
    logging.info("Analizando contenido PGN...")
    game = chess.pgn.read_game(io.StringIO(pgn_content_str))

    if not game:
        # Intentar leer múltiples juegos si el primero falla o tiene encabezado vacío
        pgn_io = io.StringIO(pgn_content_str)
        headers = chess.pgn.read_headers(pgn_io)

        if headers:
            # Si existen encabezados, intentar leer el juego después de los encabezados
            game = chess.pgn.read_game(pgn_io)

            if not game:
                raise ValueError(
                    "Archivo PGN inválido o no contiene una partida principal reconocible."
                )

    # Usar mainline_moves() para robustez
    try:
        moves = list(game.mainline_moves())
    except Exception as e:
        raise ValueError(f"Error extrayendo movimientos de la línea principal: {e}")

    if not moves:
        raise ValueError(
            "La partida PGN no contiene movimientos en la línea principal."
        )

    logging.info(f"Analizados {len(moves)} movimientos del PGN.")
    return moves


def process_pgn_file(uploaded_file):
    """Procesa un archivo PGN."""
    uploaded_file.seek(0)  # Asegurar lectura desde el inicio

    # Manejar problemas potenciales de codificación de manera más robusta
    try:
        pgn_content = uploaded_file.read().decode("utf-8-sig")  # Intentar UTF-8 primero
    except UnicodeDecodeError:
        logging.warning("Decodificación UTF-8 fallida, intentando latin-1.")
        uploaded_file.seek(0)
        pgn_content = uploaded_file.read().decode("latin-1")  # Alternativa

    st.session_state.moves = parse_pgn_content(pgn_content)  # Usar función cacheada
    st.session_state.current_move_index = -1  # Resetear índice
    st.session_state.board = chess.Board()  # Resetear tablero


def process_txt_file(uploaded_file):
    """Procesa un archivo TXT asumiendo movimientos UCI en una columna 'Moves'."""
    try:
        logging.info("Procesando archivo TXT...")

        # Asumir que load_data maneja la lectura de TXT/CSV apropiadamente
        # Pasar el objeto de archivo cargado directamente a load_data
        df = load_data(
            uploaded_file
        )  # load_data debería manejar seek(0) si es necesario

        if df is None or df.empty or "Moves" not in df.columns:
            raise ValueError(
                "Formato TXT inválido. Se espera una columna 'Moves'. Verifique data_loader.py."
            )

        # Obtener movimientos de la primera fila (ajustar si el formato difiere)
        moves_data = df.iloc[0]["Moves"]

        if isinstance(moves_data, str):
            moves_str_list = moves_data.split()
        elif isinstance(moves_data, list):
            # Manejar si ya es una lista
            moves_str_list = [str(m) for m in moves_data]
        else:
            raise ValueError(
                f"La columna 'Moves' tiene un formato inesperado: {type(moves_data)}"
            )

        logging.info(
            f"Encontrados {len(moves_str_list)} movimientos potenciales en TXT."
        )

        st.session_state.moves = []
        temp_board = chess.Board()
        invalid_moves_found = []
        processed_count = 0

        for move_str in moves_str_list:
            try:
                move = temp_board.parse_uci(
                    move_str.strip()
                )  # Eliminar espacios en blanco

                # Verificar legalidad antes de añadir
                if move in temp_board.legal_moves:
                    st.session_state.moves.append(move)
                    temp_board.push(move)
                    processed_count += 1
                else:
                    # Registrar movimientos ilegales encontrados en el archivo
                    logging.warning(
                        f"Movimiento ilegal encontrado y omitido: {move_str} en posición FEN: {temp_board.fen()}"
                    )
                    invalid_moves_found.append(move_str)
            except ValueError:
                # Registrar movimientos con formato UCI inválido
                logging.warning(f"Formato UCI inválido omitido: '{move_str}'")
                invalid_moves_found.append(move_str)

        if invalid_moves_found:
            st.warning(
                f"Se omitieron {len(invalid_moves_found)} movimientos inválidos/ilegales: {', '.join(invalid_moves_found[:5])}{'...' if len(invalid_moves_found) > 5 else ''}"
            )

        if not st.session_state.moves:
            # Verificar si la lista original tenía movimientos pero todos eran inválidos
            if moves_str_list:
                raise ValueError(
                    "No se encontraron movimientos válidos/legales en el archivo TXT."
                )
            else:
                raise ValueError("La columna 'Moves' en el archivo TXT está vacía.")

        logging.info(f"Procesados {processed_count} movimientos válidos del TXT.")
        st.session_state.current_move_index = -1  # Resetear índice
        st.session_state.board = chess.Board()  # Resetear tablero

    except Exception as e:
        # Capturar errores durante el procesamiento de TXT específicamente
        raise ValueError(f"Error procesando archivo TXT: {e}")


def analyze_loaded_game():
    """Analiza la partida cargada utilizando el evaluador y el modelo ensemble."""
    try:
        # Verificar que el evaluador esté listo
        if not st.session_state.evaluator or not hasattr(
            st.session_state.evaluator, "evaluate"
        ):
            raise ValueError("El evaluador no está inicializado correctamente.")

        with st.spinner("Analizando partida..."):
            # Inicializar variables para análisis
            board_history = []
            evaluations = []
            move_numbers = []
            critical_moves_list = []

            # Comenzar con tablero inicial
            main_board = chess.Board()
            board_history.append(main_board.copy())

            # Crear analizador
            analyzer = ChessAnalyzer(
                evaluator=st.session_state.evaluator,
                threshold=st.session_state.threshold,
                depth=1,
            )

            # Evaluar posición inicial
            try:
                initial_eval = st.session_state.evaluator.evaluate(main_board)
                evaluations.append(initial_eval)
                move_numbers.append(0)  # Posición inicial
            except Exception as e:
                logging.warning(f"Error evaluando posición inicial: {e}")
                # Usar evaluación material como respaldo
                initial_eval = st.session_state.evaluator._evaluate_material(main_board)
                evaluations.append(initial_eval)
                move_numbers.append(0)

            # Analizar cada movimiento
            for i, move in enumerate(st.session_state.moves):
                try:
                    # Aplicar movimiento
                    main_board.push(move)
                    board_history.append(main_board.copy())

                    # Evaluar nueva posición
                    try:
                        eval_score = st.session_state.evaluator.evaluate(main_board)
                        if not isinstance(eval_score, (int, float)) or math.isnan(
                            eval_score
                        ):
                            eval_score = st.session_state.evaluator._evaluate_material(
                                main_board
                            )
                    except Exception as e:
                        logging.warning(f"Error evaluando posición {i + 1}: {e}")
                        eval_score = st.session_state.evaluator._evaluate_material(
                            main_board
                        )

                    evaluations.append(eval_score)
                    move_numbers.append((i // 2) + 1 + (0.5 if i % 2 else 0))

                    # Detectar movimientos críticos
                    if i > 0:
                        eval_diff = abs(eval_score - evaluations[-2])
                        game_phase = analyzer.determine_game_phase(main_board)

                        if eval_diff > analyzer.threshold:
                            critical_moves_list.append(
                                {
                                    "move_number": (i // 2) + 1,
                                    "move": move.uci(),
                                    "eval_diff": eval_diff,
                                    "eval_before": evaluations[-2],
                                    "eval_after": eval_score,
                                    "player": "Blancas" if i % 2 == 0 else "Negras",
                                    "position_fen": main_board.fen(),
                                    "game_phase": game_phase,
                                }
                            )

                except Exception as e:
                    logging.error(f"Error procesando movimiento {i + 1}: {e}")
                    continue

            # Convertir lista a DataFrame para operaciones posteriores
            critical_moves_df = pd.DataFrame(critical_moves_list)

            # Si no hay movimientos críticos, encontrar el más significativo
            if len(critical_moves_list) == 0 and len(evaluations) > 1:
                # Encontrar el movimiento con la mayor diferencia de evaluación
                max_diff = 0
                max_diff_index = 0

                for i in range(1, len(evaluations)):
                    diff = abs(evaluations[i] - evaluations[i - 1])
                    if diff > max_diff:
                        max_diff = diff
                        max_diff_index = i

                # Añadir el movimiento más crítico aunque esté por debajo del umbral
                if max_diff > 0 and max_diff_index > 0:
                    game_phase = analyzer.determine_game_phase(
                        board_history[max_diff_index]
                    )

                    critical_moves_df = pd.DataFrame(
                        [
                            {
                                "move_number": (max_diff_index // 2) + 1,
                                "move": st.session_state.moves[
                                    max_diff_index - 1
                                ].uci(),
                                "eval_diff": max_diff,
                                "eval_before": evaluations[max_diff_index - 1],
                                "eval_after": evaluations[max_diff_index],
                                "player": "Blancas"
                                if (max_diff_index - 1) % 2 == 0
                                else "Negras",
                                "position_fen": board_history[max_diff_index].fen(),
                                "game_phase": game_phase,
                            }
                        ]
                    )

            # Filtrar movimientos críticos por fase de juego
            if not critical_moves_df.empty and st.session_state.game_phase_filter:
                critical_moves_df = critical_moves_df[
                    critical_moves_df["game_phase"].isin(
                        st.session_state.game_phase_filter
                    )
                ]

            # Predicción final con ensemble model
            try:
                raw_prediction = st.session_state.ensemble_model.predict(main_board)

                # Transformar la predicción numérica a texto descriptivo
                pred_value = raw_prediction.get("prediccion", 1)
                resultado = (
                    "Victoria de las blancas"
                    if pred_value > 1.5
                    else ("Victoria de las negras" if pred_value < 0.5 else "Tablas")
                )

                # Crear el diccionario con el formato esperado
                final_prediction = {
                    "resultado": resultado,
                    "confianza": raw_prediction.get("confianza", 0.5),
                }
            except Exception as e:
                logging.error(f"Error en predicción final: {e}")
                final_prediction = {
                    "resultado": "Error en predicción",
                    "confianza": 0.0,
                }

            # Guardar resultados en session_state
            st.session_state.board_history = board_history
            st.session_state.evaluations = evaluations
            st.session_state.move_numbers = move_numbers
            st.session_state.critical_moves = critical_moves_df
            st.session_state.final_prediction = final_prediction

            # Establecer el índice de movimiento inicial
            st.session_state.current_move_index = 0

    except Exception as e:
        st.error(f"Error durante el análisis: {e}")
        log_error("analyze_loaded_game", e)


# =============================================================================
# === Visualización de Resultados =============================================
# =============================================================================
def display_results():
    """Muestra los resultados del análisis si hay una partida cargada."""
    if not st.session_state.moves:
        return

    st.header("📊 Resultados del Análisis")

    # Crear pestañas para diferentes vistas
    tab1, tab2, tab3 = st.tabs(
        [
            "🎮 Tablero y Recomendaciones",
            "⚠️ Movimientos Críticos",
            "📈 Estadísticas y Evaluación",
        ]
    )

    with tab1:
        # Pestaña 1: Tablero, predicción, recomendaciones y mejores movimientos
        col1, col2 = st.columns([3, 2])

        with col1:
            # Mostrar tablero interactivo
            if st.session_state.board_history:
                interactive_board(
                    st.session_state.board_history, st.session_state.critical_moves
                )
            else:
                st.warning("No hay historial de tablero disponible.")

        with col2:
            # Predicción de ganador
            if hasattr(st.session_state, "final_prediction"):
                prediction = st.session_state.final_prediction
                st.markdown("### 🏆 Predicción")
                if (
                    isinstance(prediction, dict)
                    and "resultado" in prediction
                    and "confianza" in prediction
                ):
                    st.info(
                        f"**Resultado probable:** {prediction['resultado']} "
                        f"(Confianza: {prediction['confianza']:.2%})"
                    )

            # Recomendaciones estratégicas
            st.markdown("### 💡 Recomendaciones Estratégicas")
            current_board = st.session_state.board_history[
                st.session_state.current_move_index
            ]

            # Generar recomendaciones basadas en la posición actual
            if current_board:
                # Analizar estructura de peones
                pawn_structure = analyze_pawn_structure(current_board)
                st.markdown(f"**Estructura de peones:** {pawn_structure}")

                # Analizar desarrollo de piezas
                piece_development = analyze_piece_development(current_board)
                st.markdown(f"**Desarrollo de piezas:** {piece_development}")

                # Analizar seguridad del rey
                king_safety = analyze_king_safety(current_board)
                st.markdown(f"**Seguridad del rey:** {king_safety}")

            # Mejores movimientos sugeridos
            st.markdown("### ♟️ Mejores Movimientos")
            if st.session_state.evaluator and hasattr(
                st.session_state.evaluator, "evaluate"
            ):
                try:
                    # Obtener los 3 mejores movimientos para la posición actual
                    best_moves = get_best_moves(
                        current_board, st.session_state.evaluator, top_n=3
                    )

                    for i, (move, score) in enumerate(best_moves, 1):
                        st.markdown(f"**{i}.** {move} (Evaluación: {score:.3f})")
                except Exception as e:
                    st.warning(f"No se pudieron calcular los mejores movimientos: {e}")

    with tab2:
        # Pestaña 2: Movimientos críticos
        st.markdown("### ⚠️ Movimientos Críticos Detectados")

        if (
            st.session_state.critical_moves is not None
            and not st.session_state.critical_moves.empty
        ):
            display_critical_moves()
        else:
            st.info(
                "No se detectaron movimientos críticos en esta partida con el umbral actual."
            )

    with tab3:
        # Pestaña 3: Gráfico de evaluación, actividad del tablero y estadísticas
        col1, col2 = st.columns([2, 1])

        with col1:
            # Gráfico de evaluación
            st.markdown("### 📈 Gráfico de Evaluación")
            if st.session_state.evaluations and st.session_state.move_numbers:
                display_evaluation_chart()
            else:
                st.warning("No hay datos de evaluación disponibles.")

        with col2:
            # Estadísticas de la partida
            st.markdown("### 📊 Estadísticas de la Partida")
            if st.session_state.board_history:
                display_game_statistics()

        # Actividad del tablero (mapa de calor)
        st.markdown("### 🔥 Actividad del Tablero")
        if st.session_state.board_history and len(st.session_state.board_history) > 1:
            display_board_activity()
        else:
            st.warning(
                "No hay suficientes movimientos para analizar la actividad del tablero."
            )


def display_evaluation_chart():
    """Muestra un gráfico de evaluación mejorado con Plotly."""
    evaluations = st.session_state.evaluations
    move_numbers = st.session_state.move_numbers

    # Crear gráfico con Plotly
    fig = go.Figure()

    # Línea principal de evaluación
    fig.add_trace(
        go.Scatter(
            x=move_numbers,
            y=evaluations,
            mode="lines+markers",
            name="Evaluación",
            line=dict(color="blue", width=2),
            hovertemplate="Movimiento %{x}<br>Evaluación: %{y:.3f}<extra></extra>",
        )
    )

    # Marcar movimientos críticos si existen
    if (
        st.session_state.critical_moves is not None
        and not st.session_state.critical_moves.empty
    ):
        critical_x = []
        critical_y = []
        hover_texts = []

        for _, move in st.session_state.critical_moves.iterrows():
            # Encontrar el índice correspondiente en move_numbers
            move_num = move["move_number"]
            # Buscar el índice más cercano en move_numbers
            idx = min(
                range(len(move_numbers)), key=lambda i: abs(move_numbers[i] - move_num)
            )

            critical_x.append(move_numbers[idx])
            critical_y.append(evaluations[idx])
            hover_texts.append(
                f"Movimiento crítico<br>"
                f"Jugador: {move['player']}<br>"
                f"Diferencia: {move['eval_diff']:.3f}<br>"
                f"Fase: {move['game_phase']}"
            )

        fig.add_trace(
            go.Scatter(
                x=critical_x,
                y=critical_y,
                mode="markers",
                marker=dict(
                    color="red",
                    size=10,
                    symbol="star",
                    line=dict(width=2, color="DarkSlateGrey"),
                ),
                name="Movimientos Críticos",
                text=hover_texts,
                hovertemplate="%{text}<extra></extra>",
            )
        )

    # Línea de referencia en 0
    fig.add_shape(
        type="line",
        x0=min(move_numbers),
        y0=0,
        x1=max(move_numbers),
        y1=0,
        line=dict(color="gray", width=1, dash="dash"),
    )

    # Personalizar diseño
    fig.update_layout(
        title="Evaluación a lo largo de la partida",
        xaxis_title="Número de Movimiento",
        yaxis_title="Evaluación",
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=20, r=20, t=40, b=20),
    )

    # Mostrar gráfico
    st.plotly_chart(fig, use_container_width=True)


def display_critical_moves():
    """Muestra una tabla de movimientos críticos con opciones para saltar a ellos."""
    critical_moves = st.session_state.critical_moves

    # Mostrar tabla de movimientos críticos
    st.subheader("Movimientos Críticos Detectados")

    # Formatear DataFrame para mostrar
    display_df = critical_moves.copy()
    display_df = display_df[
        ["move_number", "player", "move", "eval_diff", "game_phase"]
    ]
    display_df.columns = ["Movimiento", "Jugador", "UCI", "Diferencia", "Fase"]
    display_df["Diferencia"] = display_df["Diferencia"].map(lambda x: f"{x:.3f}")

    # Añadir botón para saltar a cada movimiento
    display_df["Ver"] = range(len(display_df))

    # Mostrar tabla
    st.dataframe(display_df.set_index("Movimiento"))

    # Selector para saltar a un movimiento crítico
    if len(display_df) > 0:
        selected_critical = st.selectbox(
            "Saltar a movimiento crítico",
            options=display_df.index,
            format_func=lambda x: f"Movimiento {x} ({display_df.loc[x, 'Jugador']} - {display_df.loc[x, 'Fase']})",
        )

        if st.button("Ir a este movimiento"):
            # Encontrar el índice correspondiente en board_history
            move_num = selected_critical
            # En board_history, el índice 0 es la posición inicial, por lo que el movimiento 1 está en el índice 1
            target_index = int(move_num * 2)  # Aproximación inicial

            # Ajustar según el jugador (blancas/negras)
            player = display_df.loc[selected_critical, "Jugador"]
            if player == "Negras":
                target_index += 1

            # Asegurar que el índice esté dentro de los límites
            if 0 <= target_index < len(st.session_state.board_history):
                st.session_state.current_move_index = target_index
                st.rerun()
            else:
                st.error(f"Índice de movimiento fuera de rango: {target_index}")


def analyze_pawn_structure(board):
    """Analiza la estructura de peones y devuelve una recomendación."""
    # Contar peones doblados, aislados y pasados
    doubled_pawns = count_doubled_pawns(board)
    isolated_pawns = count_isolated_pawns(board)
    passed_pawns = count_passed_pawns(board)

    # Generar recomendación basada en el análisis
    if doubled_pawns > 2:
        return "Evita crear más peones doblados, considera reorganizar tu estructura."
    elif isolated_pawns > 2:
        return "Tienes varios peones aislados, intenta conectarlos."
    elif passed_pawns > 0:
        return "Aprovecha tus peones pasados, pueden ser decisivos en el final."
    else:
        return "Estructura sólida, mantén la cohesión de tus peones."


def analyze_piece_development(board):
    """Analiza el desarrollo de piezas y devuelve una recomendación."""
    # Verificar desarrollo de piezas menores
    developed_pieces = count_developed_pieces(board)
    total_minor_pieces = 8  # 4 piezas menores por bando

    # Generar recomendación
    if developed_pieces < 4:
        return "Prioriza el desarrollo de tus piezas menores."
    elif has_castled(board, chess.WHITE) or has_castled(board, chess.BLACK):
        return "Buen desarrollo y enroque completado, busca oportunidades tácticas."
    else:
        return "Considera completar tu desarrollo con el enroque."


def has_castled(board, color):
    """Detecta si un jugador ha enrocado basándose en la posición del rey."""
    king_square = board.king(color)
    if king_square is None:
        return False

    # Posiciones iniciales del rey
    initial_king_square = chess.E1 if color == chess.WHITE else chess.E8

    # Posiciones después de enroque
    kingside_castled_square = chess.G1 if color == chess.WHITE else chess.G8
    queenside_castled_square = chess.C1 if color == chess.WHITE else chess.C8

    # Si el rey no está en su posición inicial y está en una posición típica de enroque
    if king_square != initial_king_square and (
        king_square == kingside_castled_square
        or king_square == queenside_castled_square
    ):
        return True

    return False


def analyze_king_safety(board):
    """Analiza la seguridad del rey y devuelve una recomendación."""
    # Verificar si el rey está en jaque
    if board.is_check():
        return "¡Tu rey está en jaque! Prioriza su seguridad."

    # Verificar enroque
    if not has_castled(board, chess.WHITE) or has_castled(board, chess.BLACK):
        return "Considera enrocar pronto para mejorar la seguridad del rey."

    # Verificar peones cercanos al rey
    king_square = board.king(board.turn)
    if king_square:
        king_file = chess.square_file(king_square)
        king_rank = chess.square_rank(king_square)
        pawns_near_king = count_pawns_near_king(board, king_file, king_rank)

        if pawns_near_king < 2:
            return "La posición del rey parece vulnerable, refuerza su defensa."

    return "La posición del rey parece segura."


def get_best_moves(board, evaluator, top_n=3):
    """Calcula los mejores movimientos para la posición actual."""
    moves = list(board.legal_moves)
    move_scores = []

    for move in moves:
        # Hacer el movimiento en una copia del tablero
        board_copy = board.copy()
        board_copy.push(move)

        # Evaluar la posición resultante
        score = -evaluator.evaluate(
            board_copy
        )  # Negativo porque evaluamos desde la perspectiva del oponente
        move_scores.append((board.san(move), score))

    # Ordenar por puntuación y devolver los mejores
    move_scores.sort(key=lambda x: x[1], reverse=True)
    return move_scores[:top_n]


def count_doubled_pawns(board):
    """Cuenta los peones doblados en el tablero."""
    doubled = 0
    for file in range(8):
        white_pawns = 0
        black_pawns = 0
        for rank in range(8):
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.PAWN:
                if piece.color == chess.WHITE:
                    white_pawns += 1
                else:
                    black_pawns += 1

        doubled += max(0, white_pawns - 1) + max(0, black_pawns - 1)

    return doubled


def count_isolated_pawns(board):
    """Cuenta los peones aislados en el tablero."""
    isolated = 0
    for file in range(8):
        for color in [chess.WHITE, chess.BLACK]:
            has_pawn_in_file = False
            has_adjacent_pawn = False

            # Verificar si hay peones en esta columna
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    has_pawn_in_file = True
                    break

            # Si hay peones en esta columna, verificar columnas adyacentes
            if has_pawn_in_file:
                for adj_file in [file - 1, file + 1]:
                    if 0 <= adj_file < 8:
                        for rank in range(8):
                            square = chess.square(adj_file, rank)
                            piece = board.piece_at(square)
                            if (
                                piece
                                and piece.piece_type == chess.PAWN
                                and piece.color == color
                            ):
                                has_adjacent_pawn = True
                                break

                if not has_adjacent_pawn:
                    isolated += 1

    return isolated


def count_passed_pawns(board):
    """Cuenta los peones pasados en el tablero."""
    passed = 0
    for file in range(8):
        for color in [chess.WHITE, chess.BLACK]:
            # Dirección de avance según el color
            direction = 1 if color == chess.WHITE else -1

            # Verificar cada peón
            for rank in range(8):
                square = chess.square(file, rank)
                piece = board.piece_at(square)

                if piece and piece.piece_type == chess.PAWN and piece.color == color:
                    is_passed = True

                    # Verificar si hay peones enemigos que puedan bloquear
                    for check_file in [file - 1, file, file + 1]:
                        if 0 <= check_file < 8:
                            for check_rank in range(
                                rank + direction, 8 if direction > 0 else -1, direction
                            ):
                                if 0 <= check_rank < 8:
                                    check_square = chess.square(check_file, check_rank)
                                    check_piece = board.piece_at(check_square)
                                    if (
                                        check_piece
                                        and check_piece.piece_type == chess.PAWN
                                        and check_piece.color != color
                                    ):
                                        is_passed = False
                                        break

                    if is_passed:
                        passed += 1

    return passed


def count_developed_pieces(board):
    """Cuenta las piezas menores desarrolladas (fuera de su posición inicial)."""
    developed = 0

    # Posiciones iniciales de piezas menores (caballos y alfiles)
    initial_squares = [
        chess.B1,
        chess.G1,
        chess.C1,
        chess.F1,  # Blancas
        chess.B8,
        chess.G8,
        chess.C8,
        chess.F8,  # Negras
    ]

    # Verificar si las piezas menores están fuera de su posición inicial
    for square in initial_squares:
        piece = board.piece_at(square)
        if not piece or (
            piece.piece_type != chess.KNIGHT and piece.piece_type != chess.BISHOP
        ):
            developed += 1

    return developed


def count_pawns_near_king(board, king_file, king_rank):
    """Cuenta los peones cercanos al rey para evaluar su seguridad."""
    count = 0
    king_color = board.turn

    # Verificar casillas adyacentes al rey
    for file_offset in [-1, 0, 1]:
        for rank_offset in [-1, 0, 1]:
            if file_offset == 0 and rank_offset == 0:
                continue  # Saltar la posición del rey

            file = king_file + file_offset
            rank = king_rank + rank_offset

            if 0 <= file < 8 and 0 <= rank < 8:
                square = chess.square(file, rank)
                piece = board.piece_at(square)
                if (
                    piece
                    and piece.piece_type == chess.PAWN
                    and piece.color == king_color
                ):
                    count += 1

    return count


def display_game_statistics():
    """Muestra estadísticas generales de la partida."""
    if not st.session_state.board_history:
        st.warning("No hay datos de partida disponibles.")
        return

    # Obtener el tablero final
    final_board = st.session_state.board_history[-1]

    # Calcular estadísticas básicas
    total_moves = len(st.session_state.moves)
    captures = count_captures(st.session_state.board_history)
    checks = count_checks(st.session_state.board_history)
    material_balance = calculate_material(final_board)

    # Mostrar estadísticas
    st.metric("Total de Movimientos", total_moves)
    st.metric("Capturas", captures)
    st.metric("Jaques", checks)
    st.metric("Balance Material Final", f"{material_balance:+}")

    # Determinar fase de la partida
    game_phase = determine_game_phase(final_board)
    st.info(f"**Fase actual:** {game_phase}")


def count_captures(board_history):
    """Cuenta el número de capturas en la partida."""
    captures = 0
    for i in range(1, len(board_history)):
        prev_pieces = sum(1 for _ in board_history[i - 1].piece_map().values())
        curr_pieces = sum(1 for _ in board_history[i].piece_map().values())
        if curr_pieces < prev_pieces:
            captures += 1
    return captures


def count_checks(board_history):
    """Cuenta el número de jaques en la partida."""
    checks = 0
    for board in board_history:
        if board.is_check():
            checks += 1
    return checks


def determine_game_phase(board):
    """Determina la fase actual de la partida."""
    # Contar piezas mayores
    queens = 0
    rooks = 0
    minor_pieces = 0

    for piece in board.piece_map().values():
        if piece.piece_type == chess.QUEEN:
            queens += 1
        elif piece.piece_type == chess.ROOK:
            rooks += 1
        elif piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
            minor_pieces += 1

    # Determinar fase basada en piezas presentes
    if queens == 2 and rooks >= 3 and minor_pieces >= 4:
        return "Apertura"
    elif queens >= 1 and (rooks >= 2 or minor_pieces >= 3):
        return "Medio juego"
    else:
        return "Final"


def display_board_activity():
    """Muestra un mapa de calor de la actividad en el tablero."""
    if not st.session_state.moves or not st.session_state.board_history:
        st.warning("No hay suficientes datos para analizar la actividad del tablero.")
        return

    # Crear matriz de actividad (8x8 para el tablero)
    activity = np.zeros((8, 8))

    # Registrar cada movimiento en la matriz
    for move in st.session_state.moves:
        if hasattr(move, "from_square") and hasattr(move, "to_square"):
            from_file = chess.square_file(move.from_square)
            from_rank = chess.square_rank(move.from_square)
            to_file = chess.square_file(move.to_square)
            to_rank = chess.square_rank(move.to_square)

            activity[7 - from_rank][from_file] += 0.5  # Origen del movimiento
            activity[7 - to_rank][to_file] += 1.0  # Destino del movimiento (más peso)

    # Crear mapa de calor con Plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=activity,
            colorscale="Viridis",
            showscale=True,
            hovertemplate="Fila: %{y}<br>Columna: %{x}<br>Actividad: %{z}<extra></extra>",
        )
    )

    # Configurar ejes para mostrar notación de ajedrez
    fig.update_layout(
        title="Mapa de Actividad del Tablero",
        xaxis=dict(
            tickmode="array",
            tickvals=list(range(8)),
            ticktext=["a", "b", "c", "d", "e", "f", "g", "h"],
        ),
        yaxis=dict(
            tickmode="array",
            tickvals=list(range(8)),
            ticktext=["8", "7", "6", "5", "4", "3", "2", "1"],
        ),
    )

    st.plotly_chart(fig, use_container_width=True)


def calculate_material(board):
    """Calcula el balance material en una posición específica."""
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }
    white = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.WHITE
    )
    black = sum(
        piece_values.get(p.piece_type, 0)
        for p in board.piece_map().values()
        if p.color == chess.BLACK
    )
    return white - black


# =============================================================================
# === Función Principal =======================================================
# =============================================================================
def main():
    """Función principal que estructura la aplicación."""
    # Inicializar estado de sesión
    initialize_session_state()

    # Aplicar estilos personalizados
    apply_custom_styles()
    set_transparent_background()
    # Barra lateral para gestión de modelos
    with st.sidebar:
        st.divider()
        # Model Management section
        model_management_sidebar()
        st.divider()
        st.caption("Aplicación de Análisis de Ajedrez v1.0")  # Version info
        st.caption(f"Modelo Cargado: {'Sí' if st.session_state.model_loaded else 'No'}")
    # model_management_sidebar()

    # Título principal
    st.markdown(
        '<h1 class="main-header">♟️ Análisis Avanzado de Ajedrez</h1>',
        unsafe_allow_html=True,
    )
    st.divider()

    # Menú principal con pestañas
    tab1, tab2 = st.tabs(["🎮 Análisis de Partida", "ℹ️ Información"])

    with tab1:
        # Sistema de análisis de partidas
        game_analysis_system()

        # Mostrar resultados si hay una partida cargada
        display_results()

        # Mostrar registro de errores si hay alguno
        if st.session_state.error_log:
            with st.expander("⚠️ Registro de Errores"):
                for error in st.session_state.error_log:
                    st.error(error)

    with tab2:
        st.markdown("""
        # Análisis de Ajedrez con IA
        
        Esta aplicación utiliza modelos de aprendizaje profundo para analizar partidas de ajedrez, identificar movimientos críticos y predecir resultados.
        
        ## Características principales:
        
        - **Carga y entrena** modelos de evaluación neural
        - **Analiza partidas** en formato PGN o TXT
        - **Identifica movimientos críticos** que cambian significativamente la evaluación
        - **Visualiza la evolución** de la evaluación durante la partida
        
        
        ## Instrucciones de uso:
        
        1. Carga o entrena un modelo desde la barra lateral
        2. Sube una partida en formato PGN o TXT
        3. Presiona "Procesar y Analizar Partida"
        4. Explora los resultados y visualizaciones
        
        
        ## Créditos
        
        Desarrollado por Víctor Medrano. como parte del proyecto final de Machine Learning.
        Utiliza Python, Streamlit, python-chess y TensorFlow.
        """)


if __name__ == "__main__":
    main()
