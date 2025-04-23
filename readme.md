# Análisis Avanzado de Ajedrez

## Descripción General

Aplicación de análisis de ajedrez basada en inteligencia artificial que permite a los usuarios analizar partidas, predecir resultados y recibir recomendaciones estratégicas. La aplicación utiliza modelos de aprendizaje automático para evaluar posiciones, identificar movimientos críticos y proporcionar análisis tácticos personalizados.

## Características Principales

- **Análisis de Partidas**: Carga y analiza partidas en formato PGN o TXT.
- **Evaluación de Posiciones**: Evalúa posiciones utilizando redes neuronales entrenadas.
- **Identificación de Movimientos Críticos**: Detecta momentos decisivos en la partida.
- **Predicción de Resultados**: Predice el resultado final de una partida en curso.
- **Recomendaciones Estratégicas**: Ofrece consejos basados en la fase del juego y la posición.
- **Visualización Interactiva**: Interfaz gráfica para navegar por los movimientos de la partida.
- **Generación de Informes**: Crea informes detallados en formato Markdown y PDF.

## Estructura del Proyecto

text
ChessAI Pro/
├── ajedrez_pro.py         # Aplicación principal con interfaz avanzada
├── ajedrez_app.py         # Versión simplificada de la aplicación
├── modules/
│   ├── __init__.py        # Inicialización del paquete de módulos
│   ├── analyzer.py        # Analizador de partidas de ajedrez
│   ├── data_loader.py     # Cargador de datos de partidas
│   ├── ensemble_model.py  # Modelo ensemble para predicciones
│   ├── evaluator.py       # Evaluador de posiciones basado en redes neuronales
│   ├── model_trainer.py   # Entrenador de modelos de IA
│   ├── report_generator.py # Generador de informes y visualizaciones
│   └── utils.py           # Utilidades y funciones auxiliares
└── modelos/               # Directorio para guardar modelos entrenados

## Requisitos

Python 3.7+

Streamlit

TensorFlow

scikit-learn

XGBoost

python-chess

pandas

numpy

plotly

reportlab

cairosvg

## Instalación

Clona el repositorio:

```bash
git clone https://github.com/vmmr-d/Proyecto-de-Ajedrez.git
cd chessai-pro
```

Instala las dependencias:

```bash
pip install -r requirements.txt
```

Ejecuta la aplicación:

```bash
streamlit run ajedrez_pro.py
```

## Uso

### Análisis Básico

- Carga un modelo desde la barra lateral (o entrena uno nuevo).
- Sube un archivo PGN o TXT con una partida.
- Haz clic en "Iniciar Análisis".
- Navega por los movimientos usando los controles de navegación.
- Revisa la predicción del resultado y las recomendaciones.

### Análisis Avanzado

- Selecciona "Análisis Avanzado" en la barra lateral.
- Configura los parámetros de análisis (umbral, profundidad, etc.).
- Carga una partida y ejecuta el análisis.
- Explora los movimientos críticos identificados.
- Genera un informe detallado en formato PDF.

### Entrenamiento de Modelos

- Selecciona "Entrenamiento de Modelos" en la barra lateral.
- Carga un conjunto de datos de entrenamiento (archivo PGN con múltiples partidas).
- Configura los parámetros de entrenamiento.
- Inicia el entrenamiento.
- Guarda el modelo entrenado para uso futuro.

## Componentes Principales

### NeuralChessEvaluator

Evaluador de posiciones basado en redes neuronales que extrae características del tablero y predice la ventaja de un jugador. Incluye:

- Extracción de características posicionales.
- Modelo de red neuronal con capas densas.
- Normalización de datos con StandardScaler.
- Evaluación con búsqueda limitada (minimax).

### EnsembleModel

Modelo de conjunto que combina árboles de decisión, XGBoost y K-means para predecir el resultado de una partida. Características:

- Análisis del balance material.
- Evaluación del control del centro.
- Detección de la fase del juego.
- Generación de recomendaciones estratégicas.

### ChessAnalyzer

Analizador que identifica movimientos críticos en una partida basándose en cambios significativos en la evaluación. Funcionalidades:

- Detección de movimientos críticos con umbral adaptativo.
- Análisis por fases del juego.
- Procesamiento por lotes para partidas múltiples.
- Generación de informes detallados.

## Contribución

1. Haz un fork del repositorio.
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`).
3. Haz commit de tus cambios (`git commit -am 'Añadir nueva característica'`).
4. Haz push a la rama (`git push origin feature/nueva-caracteristica`).
5. Crea un Pull Request.

## Contacto

Para preguntas o sugerencias, por favor contacta a vmmr230812@gmail.com.