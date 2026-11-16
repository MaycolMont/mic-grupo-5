"""Alineación forzada y segmentación en fragmentos de 10-25 segundos.

Motor principal: Vosk (reconocimiento con gramática restringida al vocabulario
del ground truth) + alineación monotónica de secuencias. Vosk entrega marcas
de tiempo a nivel de palabra que se usan como anclas acústicas; las palabras
del GT que no fueron reconocidas se interpolan entre anclas. Es una alternativa
robusta en CPU cuando ``ctc-forced-aligner``/``whisperx`` no pueden instalarse
(por ejemplo, Windows sin Microsoft C++ Build Tools).

El módulo deja abierta la puerta al motor CTC: si ``ctc_forced_aligner`` está
disponible se puede solicitar con ``--engine ctc`` (pensado para GPU).
"""

from __future__ import annotations

import argparse
import difflib
import json
from collections import Counter
import math
import os
import re
import sys
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from tqdm import tqdm

try:  # permite ejecutar como script y como módulo
    from src.text_normalizer import normalizar_texto, procesar_textos, construir_mapa_clases
except ImportError:  # pragma: no cover
    from text_normalizer import normalizar_texto, procesar_textos, construir_mapa_clases

VOSK_MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-es-0.42.zip"
DEFAULT_MODEL_DIR = Path("models/vosk-model-small-es-0.42")
WORD_RE = re.compile(r"\S+", re.UNICODE)

_MODELO_VOSK = None


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
def asegurar_modelo_vosk(model_dir: str | Path = DEFAULT_MODEL_DIR) -> Path:
    """Descarga y descomprime el modelo Vosk en español si no existe."""
    model_dir = Path(model_dir)
    if model_dir.is_dir():
        return model_dir

    model_dir.parent.mkdir(parents=True, exist_ok=True)
    zip_path = model_dir.parent / f"{model_dir.name}.zip"
    if not zip_path.exists():
        import requests

        print(f"Descargando modelo Vosk desde {VOSK_MODEL_URL} ...")
        with requests.get(VOSK_MODEL_URL, stream=True, timeout=120) as respuesta:
            respuesta.raise_for_status()
            with open(zip_path, "wb") as archivo:
                for bloque in respuesta.iter_content(chunk_size=1 << 20):
                    archivo.write(bloque)

    print(f"Descomprimiendo {zip_path} ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(model_dir.parent)
    zip_path.unlink(missing_ok=True)
    if not model_dir.is_dir():
        raise FileNotFoundError(f"No se pudo preparar el modelo Vosk en {model_dir}")
    return model_dir


def _tokenizar_gt(texto_crudo: str) -> list[dict]:
    """Convierte el texto crudo en palabras GT con su forma original y normalizada.

    Conserva la puntuación adjunta en ``raw`` para reconstruir ``gt_original``.
    Si un token genera varias palabras normalizadas (p. ej. ``físico-química``),
    la forma original se asigna al primer fragmento y el resto queda vacío.
    """
    palabras: list[dict] = []
    for match in WORD_RE.finditer(texto_crudo):
        token = match.group(0)
        normalizado = normalizar_texto(token)
        if not normalizado:
            continue
        fragmentos = normalizado.split()
        for indice, fragmento in enumerate(fragmentos):
            palabras.append({"raw": token if indice == 0 else "", "norm": fragmento})
    return palabras


def _reconocer_vosk(modelo, wav_path: Path, gramatica: str | None, block_sec: int = 60) -> list[dict]:
    """Reconoce un WAV completo y devuelve palabras con tiempos absolutos."""
    from vosk import KaldiRecognizer

    with sf.SoundFile(str(wav_path)) as archivo:
        if archivo.samplerate != 16000 or archivo.channels != 1:
            raise ValueError(
                f"{wav_path.name} debe ser mono a 16 kHz "
                f"(sr={archivo.samplerate}, canales={archivo.channels})"
            )
        recognizer = (
            KaldiRecognizer(modelo, 16000, gramatica)
            if gramatica
            else KaldiRecognizer(modelo, 16000)
        )
        recognizer.SetWords(True)

        palabras: list[dict] = []
        tam_bloque = 16000 * block_sec
        while True:
            bloque = archivo.read(frames=tam_bloque, dtype="int16", always_2d=True)
            if len(bloque) == 0:
                break
            if recognizer.AcceptWaveform(bloque.tobytes()):
                resultado = json.loads(recognizer.Result())
                palabras.extend(resultado.get("result", []))
        palabras.extend(json.loads(recognizer.FinalResult()).get("result", []))

    return [
        {"word": p["word"], "norm": normalizar_texto(p["word"]), "start": float(p["start"]), "end": float(p["end"])}
        for p in palabras
        if normalizar_texto(p.get("word", ""))
    ]


def _estimar_tasa(
    palabras: list[dict],
    anclas: list[int],
    starts: list[float | None],
    ends: list[float | None],
) -> float:
    """Segundos por carácter estimados a partir de las palabras ancladas.

    Usa la mediana de las tasas por palabra para que una ancla anómala (por
    ejemplo, una palabra que Vosk estiró sobre un silencio) no contamine el
    reparto de los tramos sin anclas.
    """
    tasas: list[float] = []
    for indice in anclas:
        inicio, fin = starts[indice], ends[indice]
        if inicio is None or fin is None:
            continue
        duracion = fin - inicio
        caracteres = len(palabras[indice]["norm"])
        if 0.03 <= duracion <= 5.0 and caracteres > 0:
            tasas.append(duracion / caracteres)
    if tasas:
        return float(np.median(tasas))
    return 0.07  # ~150 ppm en español


def _tasa_hyp(palabras_hyp: list[dict]) -> float:
    """Segundos por carácter típicos según las duraciones del ASR."""
    tasas = []
    for palabra in palabras_hyp:
        duracion = palabra["end"] - palabra["start"]
        caracteres = len(palabra["norm"])
        if caracteres > 0 and 0.03 <= duracion <= 5.0:
            tasas.append(duracion / caracteres)
    return float(np.median(tasas)) if tasas else 0.07


def _ancla_plausible(
    izquierda: int,
    derecha: int,
    starts: list[float | None],
    ends: list[float | None],
    cum: np.ndarray,
    tasa: float,
) -> bool:
    """¿La transición entre dos anclas es compatible con una velocidad normal?"""
    inicio, fin = ends[izquierda], starts[derecha]
    if inicio is None or fin is None:
        return False
    dt = fin - inicio
    esperado = max((cum[derecha] - cum[izquierda]) * tasa, 0.2)
    # Cota inferior de velocidad: evita comprimir cientos de palabras en pocos
    # segundos cuando el ASR empareja una palabra con una aparición lejana.
    minimo = max(-0.5, (cum[derecha] - cum[izquierda]) * 0.03)
    return minimo <= dt <= max(esperado * 4.0, esperado + 20.0)


def _refinar_anclas(
    palabras_gt: list[dict],
    palabras_hyp: list[dict],
    starts: list[float | None],
    ends: list[float | None],
    hyp_idx: list[int | None],
    anclas: list[int],
    cum: np.ndarray,
    tasa: float,
    duracion_audio: float,
) -> list[int]:
    """Recupera anclas que el LCS global perdió por palabras repetidas.

    Para cada hueco entre anclas confiables se alinea localmente el texto GT
    contra el ASR de una ventana temporal corta. Así se evita que una palabra
    repetida se empareje con una aparición lejana en el audio.
    """
    gt_norm = [p["norm"] for p in palabras_gt]
    hyp_norm = [p["norm"] for p in palabras_hyp]
    hyp_starts = [p["start"] for p in palabras_hyp]
    hyp_ends = [p["end"] for p in palabras_hyp]

    def agregar(gt_desde: int, gt_hasta: int, j_desde: int, j_hasta: int, limite: float) -> None:
        if gt_hasta <= gt_desde or j_hasta <= j_desde:
            return
        ventana = [
            j
            for j in range(j_desde, j_hasta)
            if hyp_starts[j] <= limite
        ]
        if not ventana:
            return
        matcher = difflib.SequenceMatcher(
            None,
            gt_norm[gt_desde:gt_hasta],
            [hyp_norm[j] for j in ventana],
            autojunk=False,
        )
        for etiqueta, i1, i2, j1, j2 in matcher.get_opcodes():
            if etiqueta != "equal":
                continue
            for desplazamiento in range(i2 - i1):
                gi = gt_desde + i1 + desplazamiento
                hj = ventana[j1 + desplazamiento]
                if starts[gi] is None:
                    inicio = max(0.0, min(hyp_starts[hj], duracion_audio))
                    fin = max(inicio, min(hyp_ends[hj], duracion_audio))
                    starts[gi] = inicio
                    ends[gi] = fin
                    hyp_idx[gi] = hj

    for izquierda, derecha in zip(anclas, anclas[1:]):
        if derecha <= izquierda + 1:
            continue
        inicio = ends[izquierda]
        if inicio is None or hyp_idx[izquierda] is None or hyp_idx[derecha] is None:
            continue
        esperado = max((cum[derecha - 1] - cum[izquierda]) * tasa, 0.2)
        fin_derecha = starts[derecha] if starts[derecha] is not None else duracion_audio
        limite = min(fin_derecha, inicio + max(20.0, esperado * 3.0))
        agregar(
            izquierda + 1,
            derecha,
            hyp_idx[izquierda] + 1,
            hyp_idx[derecha],
            limite,
        )

    # Cola: texto GT posterior a la última ancla confiable.
    ultima = anclas[-1]
    if ultima < len(palabras_gt) - 1 and ends[ultima] is not None and hyp_idx[ultima] is not None:
        esperado = max((cum[-1] - cum[ultima]) * tasa, 0.2)
        limite = min(duracion_audio, ends[ultima] + max(20.0, esperado * 3.0))
        agregar(ultima + 1, len(palabras_gt), hyp_idx[ultima] + 1, len(palabras_hyp), limite)

    return [i for i, valor in enumerate(starts) if valor is not None]


def _cadena_anclas(
    palabras_gt: list[dict],
    palabras_hyp: list[dict],
    cum: np.ndarray,
    tasa: float,
) -> list[tuple[int, int, float, float]]:
    """Busca la cadena más larga de anclas temporalmente coherentes.

    Se usan solo palabras poco frecuentes (las repetidas generan emparejamientos
    ambiguos) y se encadena la coincidencia que maximiza el número de anclas
    respetando el orden del texto y una velocidad de habla plausible. Así una
    palabra repetida no arrastra la alineación a una aparición lejana.
    """
    gt_norm = [p["norm"] for p in palabras_gt]
    hyp_norm = [p["norm"] for p in palabras_hyp]
    freq_gt = Counter(gt_norm)
    umbral = 12 if len(gt_norm) >= 1000 else max(6, len(gt_norm) // 80)
    raras = {w for w, c in freq_gt.items() if c <= umbral and len(w) >= 3}

    posiciones: dict[str, list[int]] = {}
    for j, palabra in enumerate(hyp_norm):
        if palabra in raras:
            posiciones.setdefault(palabra, []).append(j)

    candidatos = [
        (i, j, palabras_hyp[j]["start"], palabras_hyp[j]["end"])
        for i, palabra in enumerate(gt_norm)
        if palabra in posiciones
        for j in posiciones[palabra]
    ]
    if not candidatos:
        return []

    candidatos.sort(key=lambda c: (c[1], c[0]))
    total = len(candidatos)
    puntaje = [1] * total
    previo = [-1] * total

    for indice, (i, _j, inicio, _fin) in enumerate(candidatos):
        mejor = 1
        enlace = -1
        for atras in range(indice - 1, -1, -1):
            pi, pj, _pinicio, pfin = candidatos[atras]
            if inicio - pfin > 900.0:
                break
            if pi >= i or pj >= _j:
                continue
            dt = inicio - pfin
            esperado = max((cum[i] - cum[pi]) * tasa, 0.2)
            minimo = max(-0.5, (cum[i] - cum[pi]) * 0.03)
            if dt < minimo or dt > max(esperado * 4.0, esperado + 30.0):
                continue
            valor = puntaje[atras] + 1
            if valor > mejor:
                mejor = valor
                enlace = atras
        puntaje[indice] = mejor
        previo[indice] = enlace

    final = max(range(total), key=puntaje.__getitem__)
    cadena: list[tuple[int, int, float, float]] = []
    while final >= 0:
        cadena.append(candidatos[final])
        final = previo[final]
    cadena.reverse()
    return cadena


def alinear_palabras(palabras_gt: list[dict], palabras_hyp: list[dict], duracion_audio: float) -> tuple[list[dict], float]:
    """Asigna tiempos a las palabras GT usando las palabras reconocidas como anclas.

    Devuelve la lista de palabras GT con ``start``/``end`` y la proporción de
    palabras GT que quedaron ancladas directamente por el reconocedor.
    """
    if not palabras_gt:
        return [], 0.0

    gt_norm = [p["norm"] for p in palabras_gt]
    hyp_norm = [p["norm"] for p in palabras_hyp]

    starts: list[float | None] = [None] * len(palabras_gt)
    ends: list[float | None] = [None] * len(palabras_gt)
    hyp_idx: list[int | None] = [None] * len(palabras_gt)
    cum = np.cumsum([max(len(p["norm"]), 1) + 1 for p in palabras_gt]).astype(float)

    tasa_hyp = _tasa_hyp(palabras_hyp)
    cadena = _cadena_anclas(palabras_gt, palabras_hyp, cum, tasa_hyp) if hyp_norm else []
    for i, j, inicio, fin in cadena:
        starts[i] = max(0.0, min(inicio, duracion_audio))
        ends[i] = max(starts[i], min(fin, duracion_audio))
        hyp_idx[i] = j

    anclas = [i for i, valor in enumerate(starts) if valor is not None]

    if not anclas:
        # Sin anclas: reparto proporcional por longitud de palabra.
        total = cum[-1] if cum[-1] > 0 else 1.0
        for i, palabra in enumerate(palabras_gt):
            palabra["start"] = duracion_audio * (cum[i] - (len(palabra["norm"]) + 1)) / total
            palabra["end"] = duracion_audio * cum[i] / total
        return palabras_gt, 0.0

    # Refina los huecos entre anclas confiables con una ventana temporal corta.
    anclas = _refinar_anclas(
        palabras_gt,
        palabras_hyp,
        starts,
        ends,
        hyp_idx,
        anclas,
        cum,
        tasa_hyp,
        duracion_audio,
    )
    filtradas = [anclas[0]]
    for indice in anclas[1:]:
        if _ancla_plausible(filtradas[-1], indice, starts, ends, cum, tasa_hyp):
            filtradas.append(indice)
    anclas = filtradas
    coincidencias = len(anclas)
    tasa = _estimar_tasa(palabras_gt, anclas, starts, ends)

    # Evita que una ancla anómala se estire sobre un silencio o un tramo no
    # transcrito: ninguna palabra debería durar decenas de segundos.
    for indice in anclas:
        duracion_max = max(1.5, len(palabras_gt[indice]["norm"]) * tasa * 4)
        if ends[indice] - starts[indice] > duracion_max:
            ends[indice] = starts[indice] + max(0.4, len(palabras_gt[indice]["norm"]) * tasa * 2)

    # Tramo inicial sin anclas: se reparte hacia atrás desde la primera ancla.
    primera = anclas[0]
    if primera > 0:
        _repartir(
            palabras_gt, cum, 0, primera, 0.0, starts[primera], tasa, alinear_fin=True
        )

    # Tramos internos entre anclas.
    for izquierda, derecha in zip(anclas, anclas[1:]):
        if derecha <= izquierda + 1:
            continue
        inicio = ends[izquierda]
        fin = starts[derecha]
        if fin < inicio:
            fin = inicio
        _repartir(palabras_gt, cum, izquierda + 1, derecha, inicio, fin, tasa)

    # Tramo final sin anclas.
    ultima = anclas[-1]
    if ultima < len(palabras_gt) - 1:
        _repartir(
            palabras_gt, cum, ultima + 1, len(palabras_gt), ends[ultima], duracion_audio, tasa
        )

    # Anclas originales y corrección de monotonía.
    previo = 0.0
    for i, palabra in enumerate(palabras_gt):
        inicio = starts[i] if starts[i] is not None else palabra.get("start", previo)
        fin = ends[i] if ends[i] is not None else palabra.get("end", inicio + 0.05)
        inicio = max(inicio, previo)
        fin = max(fin, inicio + 0.03)
        fin = min(fin, duracion_audio)
        palabra["start"] = inicio
        palabra["end"] = fin
        previo = fin

    return palabras_gt, coincidencias / max(len(palabras_gt), 1)


def _repartir(
    palabras: list[dict],
    cum: np.ndarray,
    desde: int,
    hasta: int,
    t0: float,
    t1: float,
    tasa: float,
    alinear_fin: bool = False,
) -> None:
    """Reparte palabras ``[desde, hasta)`` entre t0 y t1.

    Si el hueco es mucho mayor que la duración esperada para esas palabras
    (audio sin transcribir o silencio), se deja el excedente como pausa en vez
    de estirar las palabras. Con ``alinear_fin=True`` las palabras se anclan
    al final del intervalo (tramo inicial).
    """
    if hasta <= desde:
        return
    if t1 < t0:
        t1 = t0
    caracteres = cum[hasta - 1] - cum[desde - 1] if desde > 0 else cum[hasta - 1]
    if caracteres <= 0:
        caracteres = float(hasta - desde)
    base = cum[desde - 1] if desde > 0 else 0.0

    disponible = t1 - t0
    esperado = max(caracteres * tasa, 0.2)
    tramo = min(disponible, max(esperado * 1.5, 0.4))
    if alinear_fin:
        t0 = t1 - tramo
    t1 = t0 + tramo

    for i in range(desde, hasta):
        palabra = palabras[i]
        anterior = cum[i - 1] if i > 0 else 0.0
        palabra["start"] = t0 + (anterior - base) / caracteres * (t1 - t0)
        palabra["end"] = t0 + (cum[i] - base) / caracteres * (t1 - t0)


def construir_fragmentos(
    palabras: list[dict],
    duracion_audio: float,
    min_sec: float = 10.0,
    max_sec: float = 25.0,
    pausa_min: float = 0.35,
) -> list[list[dict]]:
    """Agrupa palabras en fragmentos de 10-25 s respetando pausas naturales."""
    fragmentos: list[list[dict]] = []
    actual: list[dict] = []

    for palabra in palabras:
        if not actual:
            actual = [palabra]
            continue
        duracion_actual = actual[-1]["end"] - actual[0]["start"]
        pausa = palabra["start"] - actual[-1]["end"]
        excederia = palabra["end"] - actual[0]["start"] > max_sec
        if (
            duracion_actual >= max_sec
            or (duracion_actual >= min_sec and pausa >= pausa_min)
            or pausa >= 5.0
            or (duracion_actual >= min_sec and excederia)
        ):
            fragmentos.append(actual)
            actual = [palabra]
        else:
            actual.append(palabra)

    if actual:
        fragmentos.append(actual)

    # Une fragmentos cortos con un vecino cuando el resultado sigue siendo
    # razonable y no hay una pausa larga de por medio.
    fusionados: list[list[dict]] = []
    for fragmento in fragmentos:
        if fusionados:
            anterior = fusionados[-1]
            duracion_anterior = anterior[-1]["end"] - anterior[0]["start"]
            duracion_fragmento = fragmento[-1]["end"] - fragmento[0]["start"]
            union = fragmento[-1]["end"] - anterior[0]["start"]
            pausa = fragmento[0]["start"] - anterior[-1]["end"]
            if (
                (duracion_anterior < min_sec or duracion_fragmento < min_sec)
                and union <= max_sec
                and pausa < 5.0
            ):
                anterior.extend(fragmento)
                continue
        fusionados.append(fragmento)

    # Filtra fragmentos sin duración válida.
    return [f for f in fusionados if f[-1]["end"] - f[0]["start"] > 0.5]


def _texto_fragmento(palabras: list[dict]) -> tuple[str, str]:
    original = " ".join(p["raw"] for p in palabras if p["raw"]).strip()
    return original, normalizar_texto(original)


# ---------------------------------------------------------------------------
# Procesamiento por clase
# ---------------------------------------------------------------------------
def _inicializar_worker(model_dir: str) -> None:
    global _MODELO_VOSK
    from vosk import Model, SetLogLevel

    SetLogLevel(-1)
    _MODELO_VOSK = Model(str(model_dir))


def procesar_clase(
    tarea: dict,
    model_dir: str,
    wav_dir: str,
    clean_dir: str,
    chunks_dir: str,
    align_dir: str,
    overwrite: bool = False,
    block_sec: int = 60,
) -> dict:
    """Alinea una clase completa y genera sus fragmentos de audio."""
    clase_id = tarea["clase_id"]
    audio_stem = tarea["audio_stem"]
    wav_path = Path(wav_dir) / f"{audio_stem}.wav"
    texto_path = Path(clean_dir) / f"{clase_id}_raw.txt"
    salida_dir = Path(chunks_dir) / clase_id
    salida_json = Path(align_dir) / f"{clase_id}.json"

    if salida_json.exists() and not overwrite:
        return {"clase_id": clase_id, "estado": "omitido", "chunks": len(json.loads(salida_json.read_text(encoding="utf-8")).get("chunks", []))}
    if not wav_path.exists():
        return {"clase_id": clase_id, "estado": "error", "detalle": f"falta {wav_path}"}
    if not texto_path.exists():
        return {"clase_id": clase_id, "estado": "error", "detalle": f"falta {texto_path}"}

    try:
        texto_crudo = texto_path.read_text(encoding="utf-8")
        palabras_gt = _tokenizar_gt(texto_crudo)
        if not palabras_gt:
            return {"clase_id": clase_id, "estado": "error", "detalle": "GT vacío"}

        info = sf.info(str(wav_path))
        duracion_audio = info.frames / info.samplerate

        vocabulario = sorted({p["norm"] for p in palabras_gt})
        gramatica = json.dumps(vocabulario + ["[unk]"], ensure_ascii=False)

        modelo = _MODELO_VOSK
        if modelo is None:  # ejecución sin ProcessPool
            from vosk import Model, SetLogLevel

            SetLogLevel(-1)
            modelo = Model(str(model_dir))

        cache_hyp = Path(align_dir) / "hyp" / f"{clase_id}.json"
        if cache_hyp.exists():
            palabras_hyp = json.loads(cache_hyp.read_text(encoding="utf-8"))
        else:
            palabras_hyp = _reconocer_vosk(modelo, wav_path, gramatica, block_sec=block_sec)
            cache_hyp.parent.mkdir(parents=True, exist_ok=True)
            cache_hyp.write_text(json.dumps(palabras_hyp, ensure_ascii=False), encoding="utf-8")

        palabras_gt, ratio = alinear_palabras(palabras_gt, palabras_hyp, duracion_audio)

        fragmentos = construir_fragmentos(palabras_gt, duracion_audio)
        salida_dir.mkdir(parents=True, exist_ok=True)
        if overwrite:
            for viejo in salida_dir.glob("*.wav"):
                viejo.unlink()

        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        chunks: list[dict] = []
        for indice, fragmento in enumerate(fragmentos, start=1):
            inicio = max(0.0, fragmento[0]["start"])
            fin = min(duracion_audio, fragmento[-1]["end"])
            if fin - inicio <= 0.5:
                continue
            sample_id = f"{clase_id}_seg{indice:04d}"
            ruta_chunk = salida_dir / f"{sample_id}.wav"
            if overwrite or not ruta_chunk.exists():
                sf.write(
                    str(ruta_chunk),
                    audio[int(inicio * sr):int(fin * sr)],
                    sr,
                    subtype="PCM_16",
                    format="WAV",
                )
            gt_original, gt_normalizado = _texto_fragmento(fragmento)
            if not gt_normalizado:
                continue
            chunks.append(
                {
                    "sample_id": sample_id,
                    "clase_id": clase_id,
                    "ruta_audio": str(ruta_chunk).replace("\\", "/"),
                    "inicio": round(inicio, 3),
                    "fin": round(fin, 3),
                    "duracion_seg": round(fin - inicio, 3),
                    "gt_original": gt_original,
                    "gt_normalizado": gt_normalizado,
                }
            )

        resultado = {
            "clase_id": clase_id,
            "audio_stem": audio_stem,
            "audio_path": str(wav_path).replace("\\", "/"),
            "duracion_audio_seg": round(duracion_audio, 3),
            "palabras_gt": len(palabras_gt),
            "palabras_reconocidas": len(palabras_hyp),
            "ratio_anclaje": round(ratio, 4),
            "chunks": chunks,
        }
        salida_json.parent.mkdir(parents=True, exist_ok=True)
        salida_json.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")

        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass

        return {
            "clase_id": clase_id,
            "estado": "ok",
            "chunks": len(chunks),
            "ratio_anclaje": round(ratio, 4),
            "palabras_gt": len(palabras_gt),
        }
    except Exception as exc:  # noqa: BLE001 - una clase no debe tumbar el lote
        import traceback

        return {
            "clase_id": clase_id,
            "estado": "error",
            "detalle": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


# ---------------------------------------------------------------------------
# Orquestación
# ---------------------------------------------------------------------------
def alinear_dataset(
    data_dir: str = "data",
    wav_dir: str = "data/interim/wav_16k",
    clean_dir: str = "data/interim/clean_texts",
    mapa_path: str = "data/interim/clase_map.json",
    chunks_dir: str = "data/processed/chunks",
    align_dir: str = "outputs/alignments",
    model_dir: str = str(DEFAULT_MODEL_DIR),
    clases: list[str] | None = None,
    workers: int = 1,
    overwrite: bool = False,
    block_sec: int = 60,
) -> list[dict]:
    """Ejecuta la alineación forzada para todas las clases del mapa."""
    mapa_path = Path(mapa_path)
    if mapa_path.exists():
        mapa = json.loads(mapa_path.read_text(encoding="utf-8"))
    else:
        mapa = construir_mapa_clases(data_dir)
        procesar_textos(data_dir, clean_dir, mapa_path)

    tareas = [fila for fila in mapa if fila.get("transcript_path")]
    if clases:
        seleccion = {c.strip() for c in clases}
        tareas = [fila for fila in tareas if fila["clase_id"] in seleccion]
    if not tareas:
        print("No hay clases con transcripción para alinear.")
        return []

    model_dir = str(asegurar_modelo_vosk(model_dir))
    Path(chunks_dir).mkdir(parents=True, exist_ok=True)
    Path(align_dir).mkdir(parents=True, exist_ok=True)

    resultados: list[dict] = []
    if workers <= 1:
        _inicializar_worker(model_dir)
        for tarea in tqdm(tareas, desc="Alineando clases", unit="clase"):
            resultado = procesar_clase(
                tarea, model_dir, wav_dir, clean_dir, chunks_dir, align_dir, overwrite, block_sec
            )
            tqdm.write(
                f"{resultado['clase_id']}: {resultado['estado']} "
                f"({resultado.get('chunks', 0)} chunks, anclaje={resultado.get('ratio_anclaje', 'n/a')})"
                + (f" -> {resultado.get('detalle')}" if resultado.get("detalle") else "")
            )
            if resultado.get("traceback"):
                tqdm.write(resultado["traceback"])
            resultados.append(resultado)
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_inicializar_worker, initargs=(model_dir,)
        ) as executor:
            futuros = {
                executor.submit(
                    procesar_clase,
                    tarea,
                    model_dir,
                    wav_dir,
                    clean_dir,
                    chunks_dir,
                    align_dir,
                    overwrite,
                    block_sec,
                ): tarea["clase_id"]
                for tarea in tareas
            }
            for futuro in tqdm(as_completed(futuros), total=len(futuros), desc="Alineando clases", unit="clase"):
                resultado = futuro.result()
                tqdm.write(
                    f"{resultado['clase_id']}: {resultado['estado']} "
                    f"({resultado.get('chunks', 0)} chunks, anclaje={resultado.get('ratio_anclaje', 'n/a')})"
                    + (f" -> {resultado.get('detalle')}" if resultado.get("detalle") else "")
                )
                if resultado.get("traceback"):
                    tqdm.write(resultado["traceback"])
                resultados.append(resultado)

    errores = [r for r in resultados if r["estado"] == "error"]
    print(
        f"Resumen alineación: {len(resultados) - len(errores)}/{len(resultados)} clases OK, "
        f"{sum(r.get('chunks', 0) for r in resultados)} fragmentos."
    )
    return resultados


def main() -> None:
    parser = argparse.ArgumentParser(description="Alineación forzada y segmentación de clases.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--wav-dir", default="data/interim/wav_16k")
    parser.add_argument("--clean-dir", default="data/interim/clean_texts")
    parser.add_argument("--map", dest="mapa_path", default="data/interim/clase_map.json")
    parser.add_argument("--chunks-dir", default="data/processed/chunks")
    parser.add_argument("--align-dir", default="outputs/alignments")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--classes", nargs="*", help="IDs de clase, p. ej. clase01 clase02")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--block-sec", type=int, default=60)
    parser.add_argument("--engine", choices=["vosk", "ctc"], default="vosk")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.engine == "ctc":
        try:
            import ctc_forced_aligner  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            print(
                "El motor CTC requiere 'ctc-forced-aligner' (y compilador C++ en Windows). "
                "Use --engine vosk.\n"
                f"Detalle: {exc}"
            )
            sys.exit(1)

    alinear_dataset(
        data_dir=args.data_dir,
        wav_dir=args.wav_dir,
        clean_dir=args.clean_dir,
        mapa_path=args.mapa_path,
        chunks_dir=args.chunks_dir,
        align_dir=args.align_dir,
        model_dir=args.model_dir,
        clases=args.classes,
        workers=args.workers,
        overwrite=args.overwrite,
        block_sec=args.block_sec,
    )


if __name__ == "__main__":
    main()
