"""Estandarización acústica de los audios originales.

Convierte todos los audios de ``data/raw`` a WAV PCM de 16 kHz, mono y con
normalización de amplitud (peak normalization). No recorta silencios para
preservar la línea temporal original del audio.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".wav"}


def estandarizar_audio(input_path: Path, output_path: Path, target_sr: int = 16000) -> float:
    """Convierte un audio a WAV mono 16 kHz normalizado en amplitud.

    Devuelve la duración en segundos del audio estandarizado.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    y, _ = librosa.load(str(input_path), sr=target_sr, mono=True)

    # Peak normalization: evita saturación y hace consistentes los niveles.
    max_val = float(np.max(np.abs(y))) if y.size else 0.0
    if max_val > 0:
        y = y / max_val

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".part")
    try:
        sf.write(str(tmp_path), y, target_sr, subtype="PCM_16", format="WAV")
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return len(y) / target_sr


def _audios_a_procesar(input_dir: Path) -> list[Path]:
    """Lista audios evitando duplicados WAV cuando ya existe el MP3 de origen."""
    archivos = [
        p
        for p in sorted(input_dir.iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    stems_mp3 = {p.stem for p in archivos if p.suffix.lower() in {".mp3", ".m4a", ".flac"}}
    return [p for p in archivos if not (p.suffix.lower() == ".wav" and p.stem in stems_mp3)]


def estandarizar_lote(
    input_dir: str | Path = "data/raw",
    output_dir: str | Path = "data/interim/wav_16k",
    target_sr: int = 16000,
    overwrite: bool = False,
) -> list[dict]:
    """Estandariza todos los audios de una carpeta.

    Omite los archivos que ya tengan su versión ``.wav`` en ``output_dir``
    (salvo que ``overwrite=True``). Nunca deja archivos parciales: escribe a
    un temporal y renombra de forma atómica.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    archivos = _audios_a_procesar(input_dir)
    if not archivos:
        print(f"No se encontraron audios en {input_dir}")
        return []

    resultados: list[dict] = []
    for archivo in tqdm(archivos, desc="Estandarizando audio", unit="archivo"):
        salida = output_dir / f"{archivo.stem}.wav"
        if salida.exists() and not overwrite:
            tqdm.write(f"Omitido (ya existe): {salida.name}")
            resultados.append(
                {"archivo": archivo.name, "salida": str(salida), "duracion_seg": None, "estado": "omitido"}
            )
            continue
        try:
            duracion = estandarizar_audio(archivo, salida, target_sr=target_sr)
            tqdm.write(f"OK  {archivo.name} -> {salida.name} ({duracion / 60:.2f} min)")
            resultados.append(
                {"archivo": archivo.name, "salida": str(salida), "duracion_seg": duracion, "estado": "ok"}
            )
        except Exception as exc:  # noqa: BLE001 - se reporta y continúa con el lote
            tqdm.write(f"ERROR {archivo.name}: {exc}")
            resultados.append(
                {"archivo": archivo.name, "salida": str(salida), "duracion_seg": None, "estado": f"error: {exc}"}
            )

    ok = sum(1 for r in resultados if r["estado"] in {"ok", "omitido"})
    print(f"Resumen: {ok}/{len(resultados)} audios disponibles en {output_dir}")
    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Estandariza audios a WAV 16 kHz mono.")
    parser.add_argument("--input", default="data/raw", help="Carpeta de audios originales.")
    parser.add_argument("--output", default="data/interim/wav_16k", help="Carpeta de salida.")
    parser.add_argument("--sr", type=int, default=16000, help="Frecuencia de muestreo objetivo.")
    parser.add_argument("--overwrite", action="store_true", help="Reprocesar audios existentes.")
    args = parser.parse_args()

    estandarizar_lote(args.input, args.output, target_sr=args.sr, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
