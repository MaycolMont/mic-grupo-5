from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".flac"}


def preprocesar_audio(input_path, output_path, target_sr=16000):
    """
    Convierte un archivo de audio a mono, reamuestrea a 16kHz,
    realiza normalización de amplitud y recorta los silencios extremos.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # 1. Cargar el audio forzando conversión a mono y la tasa de muestreo objetivo (16 kHz)
    y, sr = librosa.load(str(input_path), sr=target_sr, mono=True)

    # 2. Recorte de silencios al inicio y final (Voice Activity Detection básico por umbral)
    # top_db define el umbral en dB respecto a la amplitud máxima para considerar silencio
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)

    # 3. Normalización de amplitud (Peak Normalization o Loudness Scaling)
    # Evita saturaciones y asegura niveles consistentes de volumen entre audios
    max_val = np.max(np.abs(y_trimmed)) if y_trimmed.size else 0.0
    y_normalized = y_trimmed / max_val if max_val > 0 else y_trimmed

    # 4. Guardar el resultado optimizado en formato .wav (escritura atómica
    #    para no dejar archivos parciales si el proceso se interrumpe)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".part")
    try:
        sf.write(str(tmp_path), y_normalized, target_sr, format="WAV")
        tmp_path.replace(output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"Procesado exitosamente: {output_path.name}", flush=True)


def procesar_lote(input_dir, output_dir):
    """
    Procesa por lotes los audios de input_dir que aún no existan en output_dir.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        print(f"El directorio de entrada no existe: {input_dir}")
        return

    audios = sorted(
        p
        for p in input_dir.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audios:
        print(f"No se encontraron audios en {input_dir}")
        return

    procesados = omitidos = errores = 0
    for archivo in audios:
        out_path = output_dir / f"{archivo.stem}.wav"
        if out_path.exists():
            print(f"Omitido (ya procesado): {archivo.name}", flush=True)
            omitidos += 1
            continue
        try:
            preprocesar_audio(archivo, out_path)
            procesados += 1
        except Exception as exc:
            print(f"Error al procesar {archivo.name}: {exc}", flush=True)
            errores += 1

    print(
        f"Resumen: {procesados} procesados, {omitidos} omitidos, {errores} errores.",
        flush=True,
    )


if __name__ == "__main__":
    procesar_lote("data/raw", "data/processed")
