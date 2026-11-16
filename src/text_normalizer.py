"""Extracción y homogeneización del texto de referencia (ground truth).

- Detecta y limpia archivos WebVTT (aunque tengan extensión ``.txt``),
  descartando cabeceras y marcas de tiempo.
- Lee texto plano para los archivos sin marcas temporales.
- Ignora explícitamente las versiones ``*_gt_norm.txt`` / ``*_norm.txt`` para
  no introducir sesgos metodológicos.
- Genera una versión normalizada y determinista por clase.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import unicodedata
from pathlib import Path

TIMESTAMP_RE = re.compile(
    r"^\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})"
)
BARE_TIMESTAMP_RE = re.compile(
    r"^\s*\d{1,2}:\d{2}(?::\d{1,2})?(?:[.,]\d{1,3})?\s*$"
)
INLINE_TIMESTAMP_RE = re.compile(r"\b\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}\b")
HTML_TAG_RE = re.compile(r"<[^>]+>")
NON_VERBAL_RE = re.compile(r"\[[^\]]*\]")

# Relación entre el nombre del archivo de transcripción y el del audio.
# Se usan coincidencias por subcadena para tolerar abreviaturas y variantes.
TRANSCRIPT_TO_AUDIO = {
    "algebra-lineal": "algebra-lineal-latam",
    "calculo-vectorial": "calculo-vectorial",
    "electricidad": "electricidad-basica",
    "fisica-mecanica-ayudantias": "fisica-mecanica-ayudantias",
    "fisica-pre-de-espol": "fisica-pre-de-espol",
    "inteligencia-artificial": "inteligencia-artificial",
    "mecanica-de-maquinaria": "mecanica-de-maquinarias",
    "redes-trans-corregido": "redes-de-datos",
    "sistemas-de-control": "sistemas-de-control",
}

AUDIO_EXTENSIONS = {".mp3", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".wav"}
TRANSCRIPT_EXTENSIONS = {".vtt", ".txt", ".srt"}


def _leer_texto(ruta: Path) -> str:
    """Lee un archivo probando UTF-8, CP1252 y Latin-1 en ese orden."""
    data = Path(ruta).read_bytes()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _ts_a_segundos(valor: str) -> float:
    horas, minutos, resto = valor.replace(",", ".").split(":")
    return int(horas) * 3600 + int(minutos) * 60 + float(resto)


def _parsear_vtt(texto: str) -> tuple[str, list[dict]]:
    """Extrae texto plano y cues de un bloque WebVTT."""
    texto = texto.replace("\r\n", "\n").replace("\r", "\n")
    bloques = re.split(r"\n\s*\n", texto)
    lineas: list[str] = []
    cues: list[dict] = []

    for bloque in bloques:
        blines = [line.strip() for line in bloque.split("\n") if line.strip()]
        if not blines:
            continue

        # Cabecera WEBVTT y metadatos (Kind, Language, NOTE, STYLE, REGION).
        # Algunos archivos no separan la cabecera del primer cue con una línea
        # en blanco, por lo que se eliminan línea a línea en vez de descartar
        # el bloque completo.
        if blines[0].upper().startswith(("NOTE", "STYLE", "REGION")):
            continue
        while blines and (
            blines[0].upper().startswith("WEBVTT")
            or blines[0].upper().startswith("KIND:")
            or blines[0].upper().startswith("LANGUAGE:")
        ):
            blines.pop(0)
        if not blines:
            continue

        # Algunos bloques incluyen un identificador de cue antes del timestamp.
        indice = 0
        match = TIMESTAMP_RE.match(blines[0])
        if match is None and len(blines) > 1 and TIMESTAMP_RE.match(blines[1]):
            indice = 1
            match = TIMESTAMP_RE.match(blines[1])

        if match is None:
            contenido = blines
            inicio = fin = None
        else:
            inicio = _ts_a_segundos(match.group(1))
            fin = _ts_a_segundos(match.group(2))
            contenido = blines[indice + 1:]

        texto_cue: list[str] = []
        for linea in contenido:
            if TIMESTAMP_RE.match(linea) or BARE_TIMESTAMP_RE.match(linea):
                continue
            linea = INLINE_TIMESTAMP_RE.sub(" ", linea)
            linea = re.sub(r"\s+", " ", linea).strip()
            if not linea:
                continue
            if linea.upper().startswith(("NOTE", "STYLE", "REGION")):
                break
            # Deduplicado de subtítulos "rolling" que repiten la línea anterior.
            if lineas and lineas[-1] == linea:
                continue
            lineas.append(linea)
            texto_cue.append(linea)

        if contenido and match is not None:
            cues.append({"start": inicio, "end": fin, "text": " ".join(texto_cue)})

    return " ".join(lineas).strip(), cues


def _contenido_es_vtt(texto: str) -> bool:
    limpio = texto.lstrip("\ufeff \n\r\t")
    return limpio.upper().startswith("WEBVTT") or bool(TIMESTAMP_RE.search(limpio))


def extraer_texto_crudo(ruta_archivo: str | Path) -> str:
    """Devuelve el texto limpio de un archivo de transcripción.

    Para WebVTT (extensión ``.vtt`` o contenido que empiece con ``WEBVTT``)
    elimina cabeceras y marcas de tiempo. Para texto plano devuelve el
    contenido completo. Los archivos ``*_gt_norm.txt`` se rechazan.
    """
    ruta = Path(ruta_archivo)
    nombre = ruta.stem.lower()
    if nombre.endswith("_gt_norm") or nombre.endswith("_norm"):
        raise ValueError(f"Archivo normalizado ignorado por sesgo metodológico: {ruta.name}")

    texto = _leer_texto(ruta)
    if ruta.suffix.lower() == ".vtt" or _contenido_es_vtt(texto):
        texto_plano, _ = _parsear_vtt(texto)
        return texto_plano
    return re.sub(r"[ \t]+", " ", texto).strip()


def normalizar_texto(texto: str) -> str:
    """Normaliza de forma determinista un texto en español.

    - Minúsculas y unicode NFC.
    - Elimina etiquetas HTML, no verbales ``[Música]``, puntuación y símbolos.
    - Conserva letras (con tildes/diéresis), números y colapsa espacios.
    """
    texto = html.unescape(texto)
    texto = unicodedata.normalize("NFC", texto)
    texto = HTML_TAG_RE.sub(" ", texto)
    texto = NON_VERBAL_RE.sub(" ", texto)
    texto = texto.lower()

    limpio: list[str] = []
    for caracter in texto:
        categoria = unicodedata.category(caracter)
        if caracter.isalnum() or categoria.startswith(("L", "M", "N")):
            limpio.append(caracter)
        else:
            limpio.append(" ")

    return re.sub(r"\s+", " ", "".join(limpio)).strip()


def _descubrir_audios(data_dir: Path) -> list[Path]:
    candidatos = [
        p
        for p in sorted((data_dir / "raw").iterdir())
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]
    stems_mp3 = {p.stem for p in candidatos if p.suffix.lower() in {".mp3", ".m4a", ".flac"}}
    audios = [p for p in candidatos if not (p.suffix.lower() == ".wav" and p.stem in stems_mp3)]
    return audios


def _descubrir_transcripciones(data_dir: Path) -> list[Path]:
    """Busca transcripciones en ``data/raw`` y en carpetas ``transcripts_gt*``."""
    encontrados: dict[str, Path] = {}

    def _agregar(ruta: Path) -> None:
        nombre = ruta.stem.lower()
        if nombre.endswith("_gt_norm") or nombre.endswith("_norm"):
            return
        if ruta.suffix.lower() not in TRANSCRIPT_EXTENSIONS:
            return
        encontrados.setdefault(nombre, ruta)

    raw = data_dir / "raw"
    if raw.is_dir():
        for ruta in sorted(raw.iterdir()):
            if ruta.is_file():
                _agregar(ruta)

    for carpeta in sorted(data_dir.glob("transcripts_gt*")):
        if not carpeta.is_dir() or "norm" in carpeta.name.lower():
            continue
        for ruta in sorted(carpeta.rglob("*")):
            if ruta.is_file():
                _agregar(ruta)

    return sorted(encontrados.values(), key=lambda p: p.name.lower())


def construir_mapa_clases(data_dir: str | Path = "data") -> list[dict]:
    """Empareja cada audio con su transcripción y asigna ``clase01..claseNN``."""
    data_dir = Path(data_dir)
    audios = _descubrir_audios(data_dir)
    transcripciones = _descubrir_transcripciones(data_dir)

    mapa: list[dict] = []
    for indice, audio in enumerate(audios, start=1):
        clase_id = f"clase{indice:02d}"
        audio_min = audio.stem.lower()
        transcripcion = None
        for ruta in transcripciones:
            objetivo = TRANSCRIPT_TO_AUDIO.get(ruta.stem.lower())
            if objetivo and objetivo in audio_min:
                transcripcion = ruta
                break
        mapa.append(
            {
                "clase_id": clase_id,
                "audio_stem": audio.stem,
                "audio_path": str(audio),
                "transcript_path": str(transcripcion) if transcripcion else None,
                "transcript_stem": transcripcion.stem if transcripcion else None,
                "source_type": (
                    "vtt"
                    if transcripcion and (
                        transcripcion.suffix.lower() == ".vtt"
                        or _contenido_es_vtt(_leer_texto(transcripcion))
                    )
                    else "txt"
                )
                if transcripcion
                else None,
            }
        )

    no_emparejadas = [t.name for t in transcripciones if not any(
        TRANSCRIPT_TO_AUDIO.get(t.stem.lower(), "\0") in a["audio_stem"].lower() for a in mapa
    )]
    if no_emparejadas:
        print("Advertencia: transcripciones sin audio asociado:", ", ".join(no_emparejadas))
    for fila in mapa:
        if fila["transcript_path"] is None:
            print(f"Advertencia: sin transcripción para {fila['audio_stem']}")

    return mapa


def procesar_textos(
    data_dir: str | Path = "data",
    output_dir: str | Path = "data/interim/clean_texts",
    mapa_salida: str | Path = "data/interim/clase_map.json",
) -> list[dict]:
    """Extrae, normaliza y guarda el texto de referencia de cada clase."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    mapa = construir_mapa_clases(data_dir)
    Path(mapa_salida).parent.mkdir(parents=True, exist_ok=True)
    Path(mapa_salida).write_text(json.dumps(mapa, ensure_ascii=False, indent=2), encoding="utf-8")

    for fila in mapa:
        ruta_txt = fila["transcript_path"]
        if not ruta_txt:
            continue
        try:
            crudo = extraer_texto_crudo(ruta_txt)
            normalizado = normalizar_texto(crudo)
            if not normalizado:
                raise ValueError("el texto quedó vacío tras la normalización")

            (output_dir / f"{fila['clase_id']}.txt").write_text(normalizado, encoding="utf-8")
            (output_dir / f"{fila['clase_id']}_raw.txt").write_text(crudo, encoding="utf-8")

            # Cues solo como diagnóstico; la alineación no depende de ellos.
            if fila["source_type"] == "vtt":
                _, cues = _parsear_vtt(_leer_texto(Path(ruta_txt)))
                if cues:
                    (output_dir / f"{fila['clase_id']}_cues.json").write_text(
                        json.dumps(cues, ensure_ascii=False), encoding="utf-8"
                    )

            fila["palabras_crudas"] = len(crudo.split())
            fila["palabras_normalizadas"] = len(normalizado.split())
            print(
                f"{fila['clase_id']}: {Path(ruta_txt).name} -> "
                f"{fila['palabras_crudas']} palabras crudas, "
                f"{fila['palabras_normalizadas']} normalizadas"
            )
        except Exception as exc:  # noqa: BLE001 - se reporta y continúa
            fila["error"] = str(exc)
            print(f"ERROR {fila['clase_id']} ({Path(ruta_txt).name}): {exc}")

    Path(mapa_salida).write_text(json.dumps(mapa, ensure_ascii=False, indent=2), encoding="utf-8")
    return mapa


def main() -> None:
    parser = argparse.ArgumentParser(description="Extrae y normaliza transcripciones GT.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="data/interim/clean_texts")
    parser.add_argument("--map", dest="mapa_salida", default="data/interim/clase_map.json")
    args = parser.parse_args()
    procesar_textos(args.data_dir, args.output, args.mapa_salida)


if __name__ == "__main__":
    main()
