# Importar submódulos para facilitar su acceso desde el paquete
from .data_loader import load_data
from .evaluator import NeuralChessEvaluator
from .analyzer import ChessAnalyzer

from .model_trainer import (
    prepare_training_data,
    train_model,
    save_model,
    load_model,
)
from .ensemble_model import EnsembleModel
from .report_generator import (
    generate_report,
    generate_pdf_report,
    generate_full_report,
    generate_recommendation,
    visualize_position,
)
from .utils import log_error

# Definir __all__ para controlar qué se importa con "from modules import *"
__all__ = [
    "load_data",
    "NeuralChessEvaluator",
    "ChessAnalyzer",
    "EnsembleModel",
    "prepare_training_data",
    "train_model",
    "generate_report",
    "generate_pdf_report",
    "generate_full_report",
    "generate_recommendation",
    "visualize_position",
    "save_model",
    "load_model",
    "log_error",
]
