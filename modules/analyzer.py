import chess
import numpy as np
import pandas as pd
import streamlit as st
from modules.evaluator import NeuralChessEvaluator


class ChessAnalyzer:
    def __init__(self, evaluator, threshold=0.3, depth=1):
        """Inicializa el analizador de ajedrez con un evaluador, umbral y profundidad.

        Args:
            evaluator (NeuralChessEvaluator): Evaluador de posiciones de ajedrez.
            threshold (float): Umbral para identificar movimientos críticos.
            depth (int): Profundidad de análisis (no implementada en este código).
        """
        self.evaluator = evaluator
        self.threshold = threshold
        self.depth = depth
        self.error_log = []  # Registro de errores durante el análisis

    def analyze(self, games_df, batch_size=None):
        """Analiza partidas y encuentra movimientos críticos con umbral adaptativo.

        Args:
            games_df (pd.DataFrame o dict): DataFrame o diccionario con las partidas.
            batch_size (int, opcional): Tamaño de lote para procesamiento (no implementado).

        Returns:
            pd.DataFrame: DataFrame con los movimientos críticos encontrados.
        """
        if isinstance(games_df, dict):
            # Convertir diccionario a DataFrame si es necesario
            games_df = pd.DataFrame([games_df])

        if games_df.empty:
            # Retornar un DataFrame vacío si no hay datos
            return pd.DataFrame()

        # Asegurar que la columna "Moves" sea una cadena si contiene listas
        if isinstance(games_df, pd.DataFrame) and "Moves" in games_df.columns:
            games_df["Moves"] = games_df["Moves"].apply(
                lambda x: " ".join(x) if isinstance(x, list) else x
            )

        # Primer análisis con el umbral estándar
        critical_moves = self._process_games(games_df, batch_size=batch_size)

        # Si no se encontraron movimientos críticos, reducir el umbral
        if isinstance(critical_moves, pd.DataFrame) and critical_moves.empty:
            original_threshold = self.threshold
            self.threshold /= 2  # Reducir el umbral a la mitad temporalmente
            self.error_log.append(
                f"No se encontraron movimientos críticos con umbral {original_threshold}. Reduciendo a {self.threshold}."
            )

            critical_moves = self._process_games(games_df, batch_size=batch_size)

            # Restaurar el umbral original
            self.threshold = original_threshold

            # Si aún no hay movimientos críticos, buscar el de mayor diferencia
            if isinstance(critical_moves, pd.DataFrame) and critical_moves.empty:
                self.error_log.append(
                    f"No se encontraron movimientos críticos incluso con umbral reducido. Buscando el movimiento con mayor diferencia."
                )

                critical_moves = self._find_most_significant_moves(games_df)

        return critical_moves

    def _process_games(self, games_df, batch_size=None):
        """Procesa partidas individuales o múltiples.

        Args:
            games_df (pd.DataFrame): DataFrame con las partidas.
            batch_size (int, opcional): Tamaño de lote para procesamiento (no implementado).

        Returns:
            pd.DataFrame: DataFrame con los movimientos críticos encontrados.
        """
        if isinstance(games_df, dict):
            # Convertir diccionario a DataFrame si es necesario
            games_df = pd.DataFrame([games_df])

        all_critical_moves = []
        for _, game in games_df.iterrows():
            # Procesar cada partida individualmente
            critical_moves = self._process_single_game(game)
            all_critical_moves.extend(critical_moves)

        if not all_critical_moves:
            # Retornar un DataFrame vacío si no se encontraron movimientos críticos
            return pd.DataFrame()

        return pd.DataFrame(all_critical_moves)

    def _process_single_game(self, game):
        """Procesa una sola partida.

        Args:
            game (pd.Series o dict): Partida a procesar.

        Returns:
            list: Lista de movimientos críticos encontrados.
        """
        if isinstance(game, pd.Series):
            # Convertir Series a diccionario si es necesario
            game = game.to_dict()

        moves = game.get("Moves", "").split()  # Obtener los movimientos
        critical = []  # Lista para almacenar movimientos críticos
        board = chess.Board()  # Crear un tablero inicial

        if not self.evaluator.is_trained:
            # Registrar un error si el evaluador no está entrenado
            self.error_log.append(
                "El evaluador no está entrenado. Usando evaluación básica."
            )

        prev_eval = self.evaluator.evaluate(board)  # Evaluación inicial
        all_eval_diffs = []  # Lista para almacenar diferencias de evaluación

        for i, move_str in enumerate(moves):
            try:
                # Intentar aplicar el movimiento
                move = chess.Move.from_uci(move_str)
                board_copy = chess.Board(board.fen())
                if not board_copy.is_legal(move):
                    # Registrar un error si el movimiento no es legal
                    self.error_log.append(
                        f"Movimiento ilegal: {move_str} en posición {board.fen()}"
                    )
                    continue

                board.push(move)  # Aplicar el movimiento al tablero

                # Evaluar la nueva posición
                current_eval = self.evaluator.evaluate(board)
                eval_diff = abs(current_eval - prev_eval)
                all_eval_diffs.append((i, eval_diff, current_eval, prev_eval))

                # Determinar la fase del juego
                game_phase = self.determine_game_phase(board)

                # Registrar movimientos críticos si superan el umbral
                if eval_diff > self.threshold:
                    critical.append(
                        {
                            "move_number": i // 2 + 1 if i % 2 == 0 else (i // 2) + 1,
                            "move": move_str,
                            "eval_diff": eval_diff,
                            "eval_before": prev_eval,
                            "eval_after": current_eval,
                            "player": "Blancas" if i % 2 == 0 else "Negras",
                            "position_fen": board.fen(),
                            "game_phase": game_phase,
                            "white": game.get("White", "Desconocido"),
                            "black": game.get("Black", "Desconocido"),
                            "result": game.get("Result", "Desconocido"),
                        }
                    )

                prev_eval = current_eval  # Actualizar la evaluación previa

            except Exception as e:
                # Registrar errores durante el procesamiento de movimientos
                self.error_log.append(
                    f"Error al procesar movimiento {move_str}: {str(e)}"
                )

        # Si no hay movimientos críticos, añadir el más significativo
        if not critical and all_eval_diffs:
            all_eval_diffs.sort(key=lambda x: x[1], reverse=True)  # Ordenar por diferencia
            most_significant = all_eval_diffs[0]
            i, eval_diff, current_eval, prev_eval = most_significant

            # Recrear el tablero hasta ese punto para obtener el FEN
            temp_board = chess.Board()
            for j, move_str in enumerate(moves[: i + 1]):
                temp_board.push(chess.Move.from_uci(move_str))

            game_phase = self.determine_game_phase(temp_board)

            critical.append(
                {
                    "move_number": i // 2 + 1 if i % 2 == 0 else (i // 2) + 1,
                    "move": moves[i],
                    "eval_diff": eval_diff,
                    "eval_before": prev_eval,
                    "eval_after": current_eval,
                    "player": "Blancas" if i % 2 == 0 else "Negras",
                    "position_fen": temp_board.fen(),
                    "game_phase": game_phase,
                    "white": game.get("White", "Desconocido"),
                    "black": game.get("Black", "Desconocido"),
                    "result": game.get("Result", "Desconocido"),
                    "is_most_significant": True,
                }
            )

            self.error_log.append(
                f"No se encontraron movimientos críticos con umbral {self.threshold}. Se ha añadido el movimiento con mayor diferencia ({eval_diff:.2f})."
            )

        return critical

    def _find_most_significant_moves(self, games_df):
        """Encuentra los movimientos más significativos sin aplicar umbral.

        Args:
            games_df (pd.DataFrame o dict): DataFrame o diccionario con las partidas.

        Returns:
            pd.DataFrame: DataFrame con los movimientos más significativos.
        """
        if isinstance(games_df, dict):
            games_df = pd.DataFrame([games_df])

        significant_moves = []
        for _, game in games_df.iterrows():
            if isinstance(game, pd.Series):
                game = game.to_dict()

            moves = game.get("Moves", "").split()
            board = chess.Board()

            evaluations = [self.evaluator.evaluate(board)]

            for i, move_str in enumerate(moves):
                try:
                    move = chess.Move.from_uci(move_str)
                    if not board.is_legal(move):
                        continue

                    board.push(move)
                    evaluations.append(self.evaluator.evaluate(board))
                except Exception as e:
                    self.error_log.append(
                        f"Error al procesar movimiento {move_str}: {str(e)}"
                    )

            # Encontrar el movimiento con mayor diferencia
            max_diff = 0
            max_idx = 0

            for i in range(1, len(evaluations)):
                diff = abs(evaluations[i] - evaluations[i - 1])
                if diff > max_diff:
                    max_diff = diff
                    max_idx = i - 1  # Índice del movimiento

            # Si encontramos un movimiento significativo
            if max_diff > 0 and max_idx < len(moves):
                # Recrear el tablero hasta ese punto
                temp_board = chess.Board()
                for j in range(max_idx + 1):
                    temp_board.push(chess.Move.from_uci(moves[j]))

                game_phase = self.determine_game_phase(temp_board)

                significant_moves.append(
                    {
                        "move_number": max_idx // 2 + 1
                        if max_idx % 2 == 0
                        else (max_idx // 2) + 1,
                        "move": moves[max_idx],
                        "eval_diff": max_diff,
                        "eval_before": evaluations[max_idx],
                        "eval_after": evaluations[max_idx + 1],
                        "player": "Blancas" if max_idx % 2 == 0 else "Negras",
                        "position_fen": temp_board.fen(),
                        "game_phase": game_phase,
                        "white": game.get("White", "Desconocido"),
                        "black": game.get("Black", "Desconocido"),
                        "result": game.get("Result", "Desconocido"),
                        "is_most_significant": True,
                    }
                )

        return pd.DataFrame(significant_moves)

    def determine_game_phase(self, board):
        """Determina la fase del juego basada en el número de piezas.

        Args:
            board (chess.Board): Tablero de ajedrez.

        Returns:
            str: Fase del juego ("Apertura", "Medio juego" o "Final").
        """
        piece_count = len(board.piece_map())

        if piece_count > 24:
            return "Apertura"
        elif piece_count > 15:
            return "Medio juego"
        else:
            return "Final"

    def get_critical_moves(self, game_df):
        """Obtiene los movimientos críticos de una partida.

        Args:
            game_df (pd.DataFrame o dict): Partida a analizar.

        Returns:
            pd.DataFrame: DataFrame con los movimientos críticos.
        """
        return self.analyze(game_df)

    def get_most_critical_move(self, game_df):
        """Obtiene el movimiento más crítico de una partida.

        Args:
            game_df (pd.DataFrame o dict): Partida a analizar.

        Returns:
            dict o None: Movimiento más crítico o None si no se encuentra.
        """
        critical_moves = self.analyze(game_df)

        if isinstance(critical_moves, pd.DataFrame) and not critical_moves.empty:
            return critical_moves.loc[critical_moves["eval_diff"].idxmax()]

        return None
