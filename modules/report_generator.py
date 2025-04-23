import io
import chess
import chess.svg
import cairosvg
import pandas as pd
import streamlit as st
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from io import BytesIO


def generate_report(critical_moves, error_log, prediction):
    """Genera un reporte detallado en formato Markdown."""
    report = "# Reporte de Análisis de Ajedrez\n\n"

    # Añadir predicción del resultado
    if isinstance(prediction, dict):
        result_text = ""
        pred_value = prediction.get("prediccion", 0)
        if pred_value > 1.5:
            result_text = "Victoria de las blancas"
        elif pred_value < 0.5:
            result_text = "Victoria de las negras"
        else:
            result_text = "Tablas"

        report += f"## Predicción del Resultado Final\n"
        report += f"**Resultado predicho:** {result_text}\n"
        report += f"**Confianza:** {prediction.get('confianza', 0):.2f}\n\n"
    else:
        report += f"## Predicción del Resultado Final: {prediction}\n\n"

    # Añadir movimientos críticos
    if isinstance(critical_moves, pd.DataFrame) and not critical_moves.empty:
        report += "## Movimientos Críticos\n\n"

        # Identificar el movimiento más crítico
        most_critical_idx = (
            critical_moves["eval_diff"].idxmax()
            if "eval_diff" in critical_moves.columns
            else None
        )

        if most_critical_idx is not None:
            most_critical = critical_moves.loc[most_critical_idx]
            report += "### Movimiento Más Crítico\n\n"
            report += f"**Movimiento {most_critical['move_number']} ({most_critical['player']}):** {most_critical['move']}\n"
            report += (
                f"- **Diferencia de Evaluación:** {most_critical['eval_diff']:.2f}\n"
            )

            if "eval_before" in most_critical and "eval_after" in most_critical:
                report += (
                    f"- **Evaluación Antes:** {most_critical['eval_before']:.2f}\n"
                )
                report += (
                    f"- **Evaluación Después:** {most_critical['eval_after']:.2f}\n"
                )

            if "game_phase" in most_critical:
                report += f"- **Fase del Juego:** {most_critical['game_phase']}\n"

            report += f"- **FEN:** {most_critical['position_fen']}\n\n"

        # Añadir otros movimientos críticos
        if len(critical_moves) > 1:
            report += "### Otros Movimientos Críticos\n\n"

            for idx, move in critical_moves.iterrows():
                if most_critical_idx is not None and idx == most_critical_idx:
                    continue  # Saltar el movimiento más crítico ya mostrado

                report += f"**Movimiento {move['move_number']} ({move['player']}):** {move['move']}\n"
                report += f"- **Diferencia de Evaluación:** {move['eval_diff']:.2f}\n"

                if "eval_before" in move and "eval_after" in move:
                    report += f"- **Evaluación Antes:** {move['eval_before']:.2f}\n"
                    report += f"- **Evaluación Después:** {move['eval_after']:.2f}\n"

                if "game_phase" in move:
                    report += f"- **Fase del Juego:** {move['game_phase']}\n"

                report += f"- **FEN:** {move['position_fen']}\n\n"
    else:
        report += "No se encontraron movimientos críticos.\n\n"

    # Añadir registro de errores si es necesario
    if error_log:
        report += "## Registro de Errores\n\n"
        for error in error_log:
            report += f"- {error}\n"

    return report


def generate_pdf_report(report):
    """Genera un reporte en formato PDF con mejor formato y visualizaciones."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=18,
    )

    styles = getSampleStyleSheet()

    # Definir estilos personalizados una sola vez
    custom_styles = {
        "CustomHeading1": ParagraphStyle(
            name="CustomHeading1",
            parent=styles["Heading1"],
            spaceAfter=12,
            spaceBefore=12,
            textColor=colors.darkblue,
        ),
        "CustomHeading2": ParagraphStyle(
            name="CustomHeading2",
            parent=styles["Heading2"],
            spaceAfter=10,
            spaceBefore=10,
            textColor=colors.darkblue,
        ),
        "CustomNormal": ParagraphStyle(
            name="CustomNormal", parent=styles["Normal"], spaceAfter=8, spaceBefore=8
        ),
    }

    # Añadir estilos personalizados si no existen
    for style_name, style in custom_styles.items():
        if style_name not in styles:
            styles.add(style)

    story = []

    # Título
    story.append(Paragraph("Reporte de Análisis de Ajedrez", styles["Title"]))

    try:
        # Contenido
        paragraphs = report.split("\n\n")
        for p in paragraphs:
            if p.startswith("### "):
                # Es un subtítulo nivel 3
                story.append(
                    Paragraph(p.replace("### ", "").strip(), styles["Heading3"])
                )
            elif p.startswith("## "):
                # Es un subtítulo nivel 2
                story.append(
                    Paragraph(p.replace("## ", "").strip(), styles["CustomHeading2"])
                )
            elif p.startswith("# "):
                # Es un título
                story.append(
                    Paragraph(p.replace("# ", "").strip(), styles["CustomHeading1"])
                )
            elif p.startswith("- "):
                # Es un elemento de lista
                items = p.split("\n- ")
                for item in items:
                    if item.startswith("- "):
                        item = item[2:]
                    story.append(Paragraph(f"• {item}", styles["CustomNormal"]))
            else:
                # Es contenido normal
                story.append(Paragraph(p, styles["CustomNormal"]))
    except Exception as e:
        # Manejar errores durante el procesamiento de párrafos
        print(f"Error al procesar párrafos: {str(e)}")
        # Añadir un mensaje de error al documento
        story.append(Paragraph(f"Error al generar reporte: {str(e)}", styles["Normal"]))

    try:
        # Construir el documento
        doc.build(story)
        buffer.seek(0)
        return buffer
    except Exception as e:
        # Manejar errores durante la construcción del documento
        print(f"Error al construir el documento PDF: {str(e)}")
        # Crear un buffer con un mensaje de error
        error_buffer = io.BytesIO()
        p = SimpleDocTemplate(error_buffer, pagesize=letter)
        p.build([Paragraph(f"Error al generar el reporte: {str(e)}", styles["Normal"])])
        error_buffer.seek(0)
        return error_buffer


def visualize_position(fen, move):
    """Visualiza una posición del tablero con anotaciones mejoradas."""
    try:
        board = chess.Board(fen)
        move_obj = chess.Move.from_uci(move)

        # Añadir flechas para el movimiento
        arrows = [(move_obj.from_square, move_obj.to_square)]

        # Resaltar casillas importantes
        squares = []

        # Resaltar casillas atacadas
        for square in chess.SQUARES:
            if board.is_attacked_by(not board.turn, square):
                if (
                    board.piece_at(square)
                    and board.piece_at(square).color == board.turn
                ):
                    squares.append(square)

        svg = chess.svg.board(
            board=board, size=400, lastmove=move_obj, arrows=arrows, squares=squares
        )

        return svg
    except Exception as e:
        st.error(f"Error al visualizar la posición: {e}")
        return None


def generate_full_report(board, analyzer, ensemble_model):
    """Genera reporte unificado con predicciones y análisis táctico."""
    prediction = ensemble_model.predict(board)
    critical_move = ensemble_model.predict_critical_move(board)
    game_phase = analyzer.determine_game_phase(board)

    return {
        "prediccion": prediction,
        "movimiento_critico": critical_move,
        "fase_juego": game_phase,
        "recomendacion": generate_recommendation(board, prediction, game_phase),
    }


def generate_recommendation(board, prediction, game_phase):
    """Genera recomendaciones basadas en la posición y fase del juego."""
    recommendations = []

    # Recomendaciones basadas en la fase del juego
    if game_phase == "Apertura":
        recommendations.append("Desarrolla tus piezas y controla el centro")
        recommendations.append("Enroca pronto para proteger tu rey")
    elif game_phase == "Medio juego":
        recommendations.append("Busca oportunidades tácticas y combinaciones")
        recommendations.append("Mejora la posición de tus piezas")
    else:  # Final
        recommendations.append("Activa tu rey y avanza tus peones")
        recommendations.append("Busca oportunidades de promoción")

    # Recomendaciones basadas en la ventaja
    if isinstance(prediction, dict):
        pred_value = prediction.get("prediccion", 0)
        if pred_value > 1.5:  # Ventaja blanca
            if board.turn == chess.WHITE:
                recommendations.append(
                    "Mantén la presión y simplifica si tienes ventaja material"
                )
            else:
                recommendations.append("Busca complicaciones y contraataque")
        elif pred_value < 0.5:  # Ventaja negra
            if board.turn == chess.BLACK:
                recommendations.append(
                    "Mantén la presión y simplifica si tienes ventaja material"
                )
            else:
                recommendations.append("Busca complicaciones y contraataque")
        else:  # Posición equilibrada
            recommendations.append("Juega con precisión y busca pequeñas ventajas")

    return recommendations
