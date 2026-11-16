"""Construcción del manifiesto maestro del dataset segmentado.

Lee los metadatos de alineación de ``outputs/alignments``, verifica cada clip
(y su duración real), calcula la velocidad de habla y una estimación de SNR y
genera ``data/processed/dataset_manifest.csv``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

try:  # permite ejecutar como script y como módulo
    from src.text_normalizer import normalizar_texto
except ImportError:  # pragma: no cover
    from text_normalizer import normalizar_texto

COLUMNAS = [
    "sample_id",
    "clase_id",
    "ruta_audio",
    "duracion_seg",
    "gt_original",
    "gt_normalizado",
    "velocidad_ppm",
    "nivel_ruido_snr",
]


def estimar_snr_db(ruta_audio: str | Path, frame_ms: float = 25.0, hop_ms: float = 10.0) -> float:
    """Estima la SNR (dB) como energía de voz (p90) vs. silencio (p10)."""
    audio, sr = sf.read(str(ruta_audio), dtype="float32", always_2d=True)
    y = audio[:, 0]
    frame = max(int(sr * frame_ms / 1000), 16)
    hop = max(int(sr * hop_ms / 1000), 1)
    if y.size < frame:
        return 0.0

    num_frames = 1 + (y.size - frame) // hop
    indices = np.arange(frame)[None, :] + hop * np.arange(num_frames)[:, None]
    rms = np.sqrt(np.mean(y[indices] ** 2, axis=1))
    rms = rms[rms > 0]
    if rms.size == 0:
        return 0.0

    ruido = float(np.percentile(rms, 10))
    senal = float(np.percentile(rms, 90))
    if ruido <= 1e-9:
        return 60.0
    if senal <= ruido:
        return 0.0
    snr = 20.0 * math.log10(senal / ruido)
    return float(np.clip(snr, 0.0, 60.0))


def categoria_snr(snr: float) -> str:
    if snr < 10:
        return "Alto"
    if snr < 20:
        return "Medio"
    return "Bajo"


def construir_manifiesto(
    align_dir: str | Path = "outputs/alignments",
    salida: str | Path = "data/processed/dataset_manifest.csv",
    base_dir: str | Path = ".",
    min_duracion: float = 0.5,
    max_ppm: float = 300.0,
) -> pd.DataFrame:
    """Valida los clips alineados y escribe el manifiesto maestro."""
    align_dir = Path(align_dir)
    base_dir = Path(base_dir)
    salida = Path(salida)
    salida.parent.mkdir(parents=True, exist_ok=True)

    jsons = sorted(align_dir.glob("*.json"))
    if not jsons:
        print(f"No se encontraron alineaciones en {align_dir}")
        return pd.DataFrame(columns=COLUMNAS)

    filas: list[dict] = []
    descartados = {"duracion": 0, "texto": 0, "archivo": 0, "error": 0, "ppm_alto": 0}

    for ruta_json in tqdm(jsons, desc="Construyendo manifiesto", unit="clase"):
        try:
            datos = json.loads(ruta_json.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"ERROR leyendo {ruta_json.name}: {exc}")
            descartados["error"] += 1
            continue

        for chunk in datos.get("chunks", []):
            ruta_audio = Path(chunk.get("ruta_audio", ""))
            if not ruta_audio.is_absolute():
                ruta_audio = base_dir / ruta_audio
            if not ruta_audio.exists():
                descartados["archivo"] += 1
                continue

            try:
                info = sf.info(str(ruta_audio))
                duracion = info.frames / info.samplerate
            except Exception:  # noqa: BLE001
                descartados["error"] += 1
                continue

            if duracion <= min_duracion:
                descartados["duracion"] += 1
                continue

            gt_original = str(chunk.get("gt_original", "")).strip()
            gt_normalizado = normalizar_texto(gt_original)
            if not gt_normalizado:
                descartados["texto"] += 1
                continue

            ppm = len(gt_normalizado.split()) / (duracion / 60.0)
            if ppm > max_ppm:
                descartados["ppm_alto"] += 1
                continue

            snr = estimar_snr_db(ruta_audio)
            filas.append(
                {
                    "sample_id": chunk.get("sample_id"),
                    "clase_id": chunk.get("clase_id"),
                    "ruta_audio": ruta_audio.as_posix(),
                    "duracion_seg": round(duracion, 3),
                    "gt_original": gt_original,
                    "gt_normalizado": gt_normalizado,
                    "velocidad_ppm": round(ppm, 2),
                    "nivel_ruido_snr": round(snr, 2),
                }
            )

    manifiesto = pd.DataFrame(filas, columns=COLUMNAS)
    # Rutas relativas al directorio base para que el CSV sea portable.
    if not manifiesto.empty:
        manifiesto["ruta_audio"] = manifiesto["ruta_audio"].map(
            lambda p: str(Path(p).relative_to(base_dir.resolve()).as_posix())
            if Path(p).is_absolute() and Path(p).is_relative_to(base_dir.resolve())
            else Path(p).as_posix()
        )
        manifiesto = manifiesto.drop_duplicates(subset=["sample_id"]).reset_index(drop=True)

    manifiesto.to_csv(salida, index=False, encoding="utf-8")

    print(f"Manifiesto: {len(manifiesto)} muestras -> {salida}")
    if descartados:
        print(
            "Descartados: "
            + ", ".join(f"{clave}={valor}" for clave, valor in descartados.items() if valor)
        )
    if not manifiesto.empty:
        print(
            f"Duración total: {manifiesto['duracion_seg'].sum() / 3600:.2f} h | "
            f"ppm media: {manifiesto['velocidad_ppm'].mean():.1f} | "
            f"SNR media: {manifiesto['nivel_ruido_snr'].mean():.1f} dB"
        )
        print(
            "SNR por categoría: "
            + ", ".join(
                f"{categoria}={n}"
                for categoria, n in manifiesto["nivel_ruido_snr"].map(categoria_snr).value_counts().items()
            )
        )
    return manifiesto


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera el manifiesto maestro del dataset.")
    parser.add_argument("--align-dir", default="outputs/alignments")
    parser.add_argument("--output", default="data/processed/dataset_manifest.csv")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--min-duracion", type=float, default=0.5)
    parser.add_argument("--max-ppm", type=float, default=300.0)
    args = parser.parse_args()
    construir_manifiesto(
        args.align_dir, args.output, args.base_dir, args.min_duracion, args.max_ppm
    )


if __name__ == "__main__":
    main()
