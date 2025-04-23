import numpy as np
import os
import joblib
from tensorflow.keras.models import Sequential, load_model as keras_load_model
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
import chess


class NeuralChessEvaluator:
    def __init__(self, input_shape, initial_data=None):
        """Inicializa el evaluador de ajedrez neural.

        Args:
            input_shape: Dimensión de las características de entrada
            initial_data: Datos iniciales para entrenar el scaler (opcional)
        """
        self.model = self.build_model(input_shape)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.input_shape = input_shape

        # Inicializar el scaler con datos más representativos
        if initial_data is not None:
            # Si se proporcionan datos iniciales, usarlos para entrenar el scaler
            self.scaler.fit(initial_data)
        else:
            # Generar datos más representativos que cubran diferentes tipos de posiciones
            # Esto es mejor que datos completamente aleatorios
            dummy_data = np.zeros((20, input_shape))

            # Añadir algunas variaciones para simular diferentes posiciones
            for i in range(20):
                # Variar las proporciones de piezas (primeras 12 características)
                dummy_data[i, :12] = np.random.dirichlet(np.ones(12), size=1)[0]

                # Variar las características binarias (últimas 6)
                dummy_data[i, 12:] = np.random.randint(0, 2, size=input_shape - 12)

            self.scaler.fit(dummy_data)

        # Verificar que el scaler se haya inicializado correctamente
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            raise ValueError(
                "Error al inicializar el scaler. Verifique los datos de entrada."
            )

    def build_model(self, input_shape):
        """Construye un modelo de red neuronal para evaluación de posiciones."""
        model = Sequential(
            [
                Dense(128, activation="relu", input_shape=(input_shape,)),
                Dense(64, activation="relu"),
                Dense(32, activation="relu"),
                Dense(1, activation="tanh"),  # Tanh limita la salida entre -1 y 1
            ]
        )
        model.compile(optimizer="adam", loss="mse")
        return model

    def train(self, X, y):
        """Entrena el modelo con todos los datos a la vez."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Los datos de entrenamiento están vacíos")

        X = np.array(X)
        y = np.array(y)

        # Asegurarse de que el scaler esté entrenado
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        self.model.fit(X_scaled, y, epochs=50, validation_split=0.2, verbose=0)
        self.is_trained = True

    def train_batch(self, games, batch_size=32, epochs=5, learning_rate=0.001):
        """Entrena el modelo con un lote de datos para optimizar el uso de memoria."""
        if not games:
            return

        # Extraer características y etiquetas
        features = [self.extract_features(board) for board in games]
        y = [self._calculate_position_material(board) for board in games]

        X = np.array(features)
        y = np.array(y)

        # Ajustar el optimizador
        self.model.optimizer.lr = learning_rate

        # Asegurarse de que el scaler esté entrenado con estos datos reales
        # Esto es crucial para que las predicciones sean precisas
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # Entrenar el modelo con más verbosidad para depuración
        history = self.model.fit(
            X_scaled,
            y,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1,  # Mostrar progreso
            validation_split=0.2,  # Usar 20% para validación
        )

        self.is_trained = True

        # Devolver el historial de entrenamiento para análisis
        return history

    def predict(self, board, depth=1):
        """Predice la evaluación de una posición con una profundidad opcional."""
        if not self.is_trained:
            return self._evaluate_material(board)

        if depth <= 1:
            features = self.extract_features(board)

            # Verificar que el scaler esté entrenado
            try:
                features_scaled = self.scaler.transform(
                    np.array([features]).reshape(1, -1)
                )
                return self.model.predict(features_scaled, verbose=0)[0][0]
            except Exception as e:
                print(f"Error en predicción: {str(e)}")
                return self._evaluate_material(board)
        else:
            # Evaluación con búsqueda limitada
            return self._evaluate_with_search(board, depth)

    def _evaluate_with_search(self, board, depth):
        """Evalúa una posición considerando posibles respuestas."""
        if depth <= 0 or board.is_game_over():
            return self.predict(board, depth=0)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.predict(board, depth=0)

        # Evaluar cada movimiento posible
        evals = []
        for move in legal_moves[
            : min(5, len(legal_moves))
        ]:  # Limitar a 5 movimientos para eficiencia
            board.push(move)
            eval_score = -self._evaluate_with_search(board, depth - 1)  # Negamax
            board.pop()
            evals.append(eval_score)

        # Devolver la mejor evaluación para el jugador actual
        # Corregido: Usar max para blancas y min para negras
        if evals:
            if board.turn == chess.WHITE:
                return max(evals)  # Blancas maximizan
            else:
                return min(evals)  # Negras minimizan
        else:
            return 0.0

    def extract_features(self, board):
        """Extrae características del tablero para evaluación."""
        # Extraer características (material, posición, movilidad, etc.)
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,  # El rey no tiene valor en este contexto
        }

        # Características de piezas (12 primeras)
        features = np.zeros(12)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                idx = piece.piece_type - 1 + (6 if piece.color == chess.BLACK else 0)
                features[idx] += 1

        # Normalizar el número de piezas
        total_pieces = np.sum(features)
        if total_pieces > 0:
            features /= total_pieces

        # Características adicionales
        additional_features = np.array(
            [
                int(board.turn),
                int(board.has_kingside_castling_rights(chess.WHITE)),
                int(board.has_queenside_castling_rights(chess.WHITE)),
                int(board.has_kingside_castling_rights(chess.BLACK)),
                int(board.has_queenside_castling_rights(chess.BLACK)),
                int(board.is_check()),
            ]
        )

        # Combinar todas las características
        all_features = np.concatenate([features, additional_features])

        # Normalizar todas las características juntas para mantener la escala adecuada
        # Usamos una normalización min-max para mantener los valores entre 0 y 1
        max_val = (
            np.max(np.abs(all_features)) if np.max(np.abs(all_features)) != 0 else 1
        )
        all_features = all_features / max_val

        return all_features

    def evaluate(self, board, depth=1):
        """Evalúa una posición y devuelve un valor entre -1 y 1."""
        try:
            if not self.is_trained:
                return self._evaluate_material(board)

            # Si se solicita búsqueda en profundidad, usar la función correspondiente
            if depth > 1:
                return self._evaluate_with_search(board, depth)

            # Evaluación directa sin búsqueda
            features = self.extract_features(board)

            # Verificar que las características sean válidas
            if features is None or len(features) != self.input_shape:
                print(
                    f"Error: Características inválidas (longitud {len(features) if features is not None else 'None'}, esperada {self.input_shape})"
                )
                return self._evaluate_material(board)

            # Verificar que el scaler esté entrenado
            if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
                print("Error: Scaler no entrenado correctamente")
                return self._evaluate_material(board)

            # Transformar y predecir
            features_scaled = self.scaler.transform([features])
            evaluation = self.model.predict(features_scaled, verbose=0)[0][0]

            # Validar rango de salida
            if not (-1 <= evaluation <= 1):
                print(
                    f"Advertencia: Evaluación fuera de rango ({evaluation}), usando evaluación material"
                )
                return self._evaluate_material(board)

            return evaluation

        except Exception as e:
            print(f"Error en evaluación: {str(e)}")
            # En caso de error, usar la evaluación material como respaldo
            return self._evaluate_material(board)

    def _evaluate_material(self, board):
        """Evaluación simple basada en material."""
        piece_values = {
            chess.PAWN: 1,
            chess.KNIGHT: 3,
            chess.BISHOP: 3,
            chess.ROOK: 5,
            chess.QUEEN: 9,
            chess.KING: 0,
        }

        white_material = sum(
            piece_values.get(p.piece_type, 0)
            for p in board.piece_map().values()
            if p.color == chess.WHITE
        )

        black_material = sum(
            piece_values.get(p.piece_type, 0)
            for p in board.piece_map().values()
            if p.color == chess.BLACK
        )

        # Normalizar a un rango entre -1 y 1
        diff = white_material - black_material
        max_material = 39  # 9 (reina) + 2*5 (torres) + 2*3 (alfiles) + 2*3 (caballos) + 8*1 (peones)
        return max(min(diff / max_material, 1.0), -1.0)

    def _calculate_position_material(self, board):
        """Método de cálculo de material (ya existente con otro nombre)"""
        return self._evaluate_material(board)  # Reutilizar implementación existente

    def save(self, filepath):
        """Guarda el modelo y el scaler en un archivo."""
        model_path = filepath + "_model.h5"
        scaler_path = filepath + "_scaler.joblib"

        # Crear directorio si no existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Guardar modelo y scaler
        self.model.save(model_path)

        # Verificar que el scaler esté entrenado antes de guardarlo
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            # Entrenar con datos dummy si es necesario
            dummy_data = np.random.rand(10, self.input_shape)
            self.scaler.fit(dummy_data)

        joblib.dump(self.scaler, scaler_path)

        # Guardar metadatos
        metadata = {"input_shape": self.input_shape, "is_trained": self.is_trained}
        joblib.dump(metadata, filepath + "_metadata.joblib")

        return model_path, scaler_path

    def load(self, filepath):
        """Carga el modelo y el scaler desde un archivo."""
        model_path = filepath + "_model.h5"
        scaler_path = filepath + "_scaler.joblib"
        metadata_path = filepath + "_metadata.joblib"

        # Cargar modelo
        self.model = keras_load_model(model_path)

        # Cargar scaler
        self.scaler = joblib.load(scaler_path)

        # Verificar que el scaler esté correctamente cargado
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            # Si el scaler no está correctamente cargado, inicializarlo con datos dummy
            dummy_data = np.random.rand(10, self.input_shape)
            self.scaler.fit(dummy_data)
            print(
                "Advertencia: Scaler no cargado correctamente, inicializado con datos dummy"
            )

        # Cargar metadatos
        metadata = joblib.load(metadata_path)
        self.input_shape = metadata["input_shape"]
        self.is_trained = metadata["is_trained"]

        return self

    def batch_predict(self, boards, batch_size=32):
        """Realiza predicciones en lotes para múltiples posiciones."""
        if not self.is_trained:
            return [self._evaluate_material(board) for board in boards]

        features = [self.extract_features(board) for board in boards]
        features = np.array(features)

        # Verificar que el scaler esté entrenado
        if not hasattr(self.scaler, "mean_") or not hasattr(self.scaler, "scale_"):
            return [self._evaluate_material(board) for board in boards]

        # Procesar por lotes para evitar problemas de memoria
        predictions = []
        for i in range(0, len(features), batch_size):
            batch = features[i : i + batch_size]
            try:
                batch_scaled = self.scaler.transform(batch)
                batch_predictions = self.model.predict(
                    batch_scaled, verbose=0
                ).flatten()
                predictions.extend(batch_predictions)
            except Exception as e:
                print(f"Error en predicción por lotes: {str(e)}")
                # Usar evaluación material como respaldo para este lote
                for j in range(i, min(i + batch_size, len(boards))):
                    predictions.append(self._evaluate_material(boards[j]))

        return predictions
