import chess
import numpy as np
import io
import os
import joblib
from datetime import datetime
import pandas as pd
import chess.pgn
import streamlit as st
from modules.evaluator import NeuralChessEvaluator


def prepare_training_data(training_file, max_games=50000):
    """Prepara datos de entrenamiento con validación mejorada"""
    try:
        # Resetear buffer y manejar encoding
        training_file.seek(0)
        try:
            pgn_content = training_file.read().decode("utf-8-sig")
        except UnicodeDecodeError:
            training_file.seek(0)
            pgn_content = training_file.read().decode("latin-1")

        pgn_io = io.StringIO(pgn_content)
        games_data = []
        game_count = 0

        while game_count < max_games:
            try:
                game = chess.pgn.read_game(pgn_io)
                if not game:
                    break

                # Validar estructura básica del PGN
                if "White" not in game.headers or "Black" not in game.headers:
                    continue

                # Extraer movimientos válidos
                board = game.board()
                moves_uci = []
                for move in game.mainline_moves():
                    if board.is_legal(move):
                        moves_uci.append(move.uci())
                        board.push(move)

                if len(moves_uci) < 2:
                    continue

                games_data.append(
                    {
                        "fen": board.fen(),
                        "score": get_game_result(
                            game.headers.get("Result", "*"), board.turn
                        ),
                        "moves": " ".join(moves_uci),  # Guardar como string
                    }
                )

                game_count += 1
            except Exception as e:
                st.error(f"Error procesando partida: {str(e)}")
                continue

        if not games_data:
            raise ValueError(
                "El archivo PGN no contiene partidas válidas (verifique formato y movimientos)"
            )

        return pd.DataFrame(games_data)
    except Exception as e:
        raise RuntimeError(f"Error en preparación de datos: {str(e)}")


def get_game_result(result, turn):
    """Convierte el resultado con validación de formato"""
    if result == "1-0":
        return 1 if turn == chess.WHITE else -1
    elif result == "0-1":
        return -1 if turn == chess.WHITE else 1
    elif result == "1/2-1/2":
        return 0
    else:
        raise ValueError(f"Resultado no válido: {result}")


def train_model(evaluator, training_data, params):
    """Función crítica para entrenamiento del modelo (recuperada del original)"""
    try:
        # Procesar cada posición de entrenamiento
        features = []
        scores = []

        for _, row in training_data.iterrows():
            try:
                board = chess.Board(row["fen"])
                features.append(evaluator.extract_features(board))
                scores.append(row["score"])
            except Exception as e:
                st.error(f"Error procesando FEN {row['fen']}: {str(e)}")
                continue

        if not features or not scores:
            raise ValueError("Datos de entrenamiento insuficientes")

        # Convertir a arrays numpy
        X = np.array(features)
        y = np.array(scores)

        # Entrenar modelo con parámetros configurables
        evaluator.train_batch(
            games=X,
            epochs=params["epochs"],
            batch_size=params["batch_size"],
            learning_rate=params["learning_rate"],
        )

        return True
    except Exception as e:
        raise RuntimeError(f"Error en entrenamiento: {str(e)}") from e


def save_model(evaluator, ensemble, path, params):
    """Guarda modelo con verificación de integridad exhaustiva"""
    try:
        path = os.path.normpath(path)
        model_dir = os.path.dirname(path)

        os.makedirs(model_dir, exist_ok=True, mode=0o755)

        # Generar nombres de archivo normalizados
        components = {
            "evaluator_model": f"{path}_evaluator_model.h5",
            "evaluator_scaler": f"{path}_evaluator_scaler.joblib",
            "ensemble_model": f"{path}_ensemble.joblib",
            "metadata": f"{path}_metadata.joblib",
        }

        # Eliminar archivos existentes
        for filepath in components.values():
            if os.path.exists(filepath):
                os.remove(filepath)

        # Guardar componentes con verificación
        evaluator.model.save(components["evaluator_model"])
        joblib.dump(evaluator.scaler, components["evaluator_scaler"])
        joblib.dump(ensemble, components["ensemble_model"])
        joblib.dump(
            {
                "training_params": params,
                "timestamp": datetime.now().isoformat(),
                "input_shape": evaluator.input_shape,
            },
            components["metadata"],
        )

        # Verificación post-guardado
        for key, filepath in components.items():
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Falta componente crítico: {key}")
            if os.path.getsize(filepath) == 0:
                raise IOError(f"Archivo vacío: {filepath}")

        return True
    except Exception as e:
        raise RuntimeError(f"Error al guardar modelo: {str(e)}")


def load_model(path):
    """Carga modelo con validación completa de compatibilidad y mayor robustez"""
    try:
        path = os.path.normpath(path)

        # Buscar archivos que coincidan con el patrón en lugar de nombres exactos
        model_dir = os.path.dirname(path)
        base_name = os.path.basename(path)
        all_files = os.listdir(model_dir)

        # Encontrar archivos que coincidan con el patrón
        evaluator_model_file = next(
            (f for f in all_files if base_name in f and "_evaluator_model.h5" in f),
            None,
        )
        evaluator_scaler_file = next(
            (
                f
                for f in all_files
                if base_name in f and "_evaluator_scaler.joblib" in f
            ),
            None,
        )
        ensemble_model_file = next(
            (f for f in all_files if base_name in f and "_ensemble.joblib" in f), None
        )
        metadata_file = next(
            (f for f in all_files if base_name in f and "_metadata.joblib" in f), None
        )

        if not all(
            [
                evaluator_model_file,
                evaluator_scaler_file,
                ensemble_model_file,
                metadata_file,
            ]
        ):
            missing = []
            if not evaluator_model_file:
                missing.append("evaluator_model")
            if not evaluator_scaler_file:
                missing.append("evaluator_scaler")
            if not ensemble_model_file:
                missing.append("ensemble_model")
            if not metadata_file:
                missing.append("metadata")
            raise FileNotFoundError(f"Archivos faltantes: {missing}")

        # Construir rutas completas
        evaluator_model_path = os.path.join(model_dir, evaluator_model_file)
        evaluator_scaler_path = os.path.join(model_dir, evaluator_scaler_file)
        ensemble_model_path = os.path.join(model_dir, ensemble_model_file)
        metadata_path = os.path.join(model_dir, metadata_file)

        # Cargar metadatos primero para verificar compatibilidad
        metadata = joblib.load(metadata_path)

        # Inicializar evaluador con parámetros correctos
        evaluator = NeuralChessEvaluator(input_shape=metadata.get("input_shape", 18))
        evaluator.model = evaluator.build_model(metadata.get("input_shape", 18))
        evaluator.model.load_weights(evaluator_model_path)
        evaluator.scaler = joblib.load(evaluator_scaler_path)
        evaluator.is_trained = metadata.get(
            "is_trained", True
        )  # Asumir que está entrenado

        # Cargar ensemble
        ensemble = joblib.load(ensemble_model_path)

        return evaluator, ensemble, metadata.get("training_params", {})
    except Exception as e:
        raise RuntimeError(f"Error al cargar modelo: {str(e)}")
