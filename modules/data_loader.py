import pandas as pd
import chess.pgn
import io
import streamlit as st

def load_data(uploaded_file):
    """Carga datos con manejo robusto de encoding y reset de buffer.
    
    Args:
        uploaded_file (UploadedFile): Archivo cargado por el usuario.
    
    Returns:
        pd.DataFrame: DataFrame con los datos procesados del archivo PGN.
        None: Si no se puede cargar el archivo o no es válido.
    """
    if uploaded_file is None:
        # Si no se proporciona un archivo, devolver None
        return None
    
    try:
        # Resetear el buffer antes de cada lectura para evitar errores de posición
        uploaded_file.seek(0)
        
        if uploaded_file.name.endswith(".pgn"):
            # Manejar diferentes encodings para archivos PGN
            try:
                # Intentar decodificar como UTF-8 con firma BOM
                pgn_content = uploaded_file.read().decode("utf-8-sig")
            except UnicodeDecodeError:
                # Si falla, intentar con encoding Latin-1
                uploaded_file.seek(0)
                pgn_content = uploaded_file.read().decode("latin-1")
                
            # Parsear el contenido PGN a un DataFrame
            return parse_pgn_to_dataframe(pgn_content)
        else:
            # Lanzar un error si el formato del archivo no es soportado
            raise ValueError("Formato no soportado")
    except Exception as e:
        # Mostrar un mensaje de error en la interfaz de Streamlit
        st.error(f"Error al cargar archivo: {str(e)}")
        return None

def parse_pgn_to_dataframe(pgn_text):
    """Parsea el contenido de un archivo PGN y lo convierte en un DataFrame.
    
    Args:
        pgn_text (str): Contenido del archivo PGN como texto.
    
    Returns:
        pd.DataFrame: DataFrame con las partidas procesadas.
    
    Raises:
        ValueError: Si no se encuentran partidas válidas en el archivo.
    """
    games = []  # Lista para almacenar las partidas procesadas
    pgn = io.StringIO(pgn_text)  # Crear un buffer de texto para leer el PGN
    
    while True:
        try:
            # Leer una partida del archivo PGN
            game = chess.pgn.read_game(pgn)
            if not game:
                # Si no hay más partidas, salir del bucle
                break
            
            # Extraer movimientos válidos de la partida
            board = game.board()
            moves_uci = []  # Lista para almacenar los movimientos en formato UCI
            for move in game.mainline_moves():
                if board.is_legal(move):
                    # Si el movimiento es legal, agregarlo a la lista
                    moves_uci.append(move.uci())
                    board.push(move)  # Aplicar el movimiento al tablero
            
            if len(moves_uci) < 2:
                # Ignorar partidas con menos de 2 movimientos
                continue
            
            # Crear una entrada con los datos de la partida
            games.append({
                "Moves": " ".join(moves_uci),  # Almacenar los movimientos como una cadena
                "White": game.headers.get("White", "Desconocido"),  # Nombre del jugador con blancas
                "Black": game.headers.get("Black", "Desconocido"),  # Nombre del jugador con negras
                "Result": game.headers.get("Result", "*"),  # Resultado de la partida
                "Event": game.headers.get("Event", ""),  # Evento de la partida
                "ECO": game.headers.get("ECO", "")  # Código ECO de la apertura
            })
            
        except Exception as e:
            # Ignorar errores y continuar con la siguiente partida
            continue
    
    if not games:
        # Si no se encontraron partidas válidas, lanzar un error
        raise ValueError("Archivo no contiene partidas válidas")
    
    # Convertir la lista de partidas en un DataFrame de Pandas
    return pd.DataFrame(games)
