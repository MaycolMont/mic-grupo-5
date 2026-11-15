# ASR Research Project

Evaluación comparativa de modelos ASR (Whisper, Vosk, entre otros) sobre audios de clase.

## Estructura

- `data/raw`: audios originales (`.mp3`) y transcripciones crudas.
- `data/processed`: audios preprocesados (16 kHz, mono, `.wav`).
- `notebooks/`: exploración y pruebas piloto.
- `src/`: código modular (preprocesamiento, chunking, inferencia, evaluación).
- `outputs/hypotheses`: transcripciones predichas por cada modelo.
- `outputs/metrics`: reportes CSV/JSON con WER, CER y estadísticas.
- `tests/`: pruebas unitarias.

## Instalación

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
