"""
=========================================================
TRADUCTOR DE VIDEOS IA — V5.2 (CODE-SWITCHING JA/EN)
=========================================================
Genera subtitulos .srt traducidos al espanol.
Si el idioma es japones, genera ademas:
  - .srt con texto original (kanji/kana)
  - .srt con Romaji (transliteracion fonetica)

Detecta automaticamente cuando el vocalista cambia
de japones a ingles dentro de la misma cancion.

Dependencias extra para Romaji:
    pip install cutlet unidic-lite
=========================================================
"""

import os
import sys
import unicodedata
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Intentar importar cutlet (Romaji) ──────────────────
try:
    import cutlet
    ROMAJI_DISPONIBLE = True
except ImportError:
    ROMAJI_DISPONIBLE = False

# ── CONFIGURACION ──────────────────────────────────────
MODELO_WHISPER = "medium"          # "small" | "medium" | "large-v3"
IDIOMA_DESTINO = "spa_Latn"        # Codigo NLLB del idioma destino
MODELO_NLLB    = "facebook/nllb-200-distilled-600M"

# Mapa Whisper -> NLLB
MAPA_IDIOMAS = {
    "en": "eng_Latn", "ja": "jpn_Jpan", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ko": "kor_Hang",
    "nl": "nld_Latn", "tr": "tur_Latn", "ar": "arb_Arab",
    "sv": "swe_Latn", "pl": "pol_Latn", "hi": "hin_Deva",
    "es": "spa_Latn", "id": "ind_Latn", "vi": "vie_Latn",
}

# Nombres legibles para el menu
NOMBRES_IDIOMAS = {
    "en": "Ingles",    "ja": "Japones",    "fr": "Frances",
    "de": "Aleman",    "it": "Italiano",   "pt": "Portugues",
    "ru": "Ruso",      "zh": "Chino",      "ko": "Coreano",
    "nl": "Holandes",  "tr": "Turco",      "ar": "Arabe",
    "sv": "Sueco",     "pl": "Polaco",     "hi": "Hindi",
    "es": "Espanol",   "id": "Indonesio",  "vi": "Vietnamita",
}

EXTENSIONES_VIDEO = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}

# Prompt inicial en japones para guiar a Whisper sin forzarlo.
# Esto sesga el modelo hacia japones pero le permite detectar
# ingles cuando aparece. Contiene vocabulario musical comun.
PROMPT_JAPONES = (
    "音楽、歌詞、日本語と英語が混ざった歌。"
    "ロック、メタル、ビジュアル系。"
)


# ── DETECCION DE IDIOMA POR SEGMENTO ──────────────────
def detectar_idioma_texto(texto):
    """
    Analiza los caracteres Unicode de un segmento para determinar
    si es japones o ingles (u otro idioma latino).

    Logica:
      - Caracteres Hiragana: U+3040 a U+309F  (ej: あいうえお)
      - Caracteres Katakana: U+30A0 a U+30FF  (ej: アイウエオ)
      - Kanji (CJK):        U+4E00 a U+9FFF  (ej: 漢字)
      - Katakana extendido:  U+31F0 a U+31FF
      - Si mas del 15% de los caracteres son japoneses -> "ja"
      - Si no -> "en" (o el idioma latino que sea)

    Por que 15%? Porque en frases japonesas cortas los espacios
    y puntuacion reducen el porcentaje. 15% es suficiente para
    detectar incluso una sola palabra en kanji/kana mezclada
    con puntuacion.

    Retorna: "ja" o "en"
    """
    if not texto:
        return "en"

    total = 0
    japones = 0

    for char in texto:
        if char in " \t\n":
            continue
        total += 1
        cp = ord(char)
        # Hiragana
        if 0x3040 <= cp <= 0x309F:
            japones += 1
        # Katakana
        elif 0x30A0 <= cp <= 0x30FF:
            japones += 1
        # Kanji (CJK Unified Ideographs)
        elif 0x4E00 <= cp <= 0x9FFF:
            japones += 1
        # Katakana extendido
        elif 0x31F0 <= cp <= 0x31FF:
            japones += 1
        # CJK Extension A
        elif 0x3400 <= cp <= 0x4DBF:
            japones += 1
        # Simbolos CJK y puntuacion japonesa
        elif 0x3000 <= cp <= 0x303F:
            japones += 1

    if total == 0:
        return "en"

    ratio = japones / total
    return "ja" if ratio > 0.15 else "en"


# ── MENU INTERACTIVO ──────────────────────────────────
def mostrar_menu_idioma():
    """Muestra menu para elegir idioma o auto-detectar."""
    print("\n  Que idioma tiene el video?\n")
    print("   [0] Auto-detectar (dejar que la IA adivine)")

    codigos = list(MAPA_IDIOMAS.keys())
    for i, cod in enumerate(codigos, 1):
        nombre = NOMBRES_IDIOMAS.get(cod, cod)
        marca = " ***" if cod == "ja" else ""
        print("   [{:>2}] {} ({}){}".format(i, nombre, cod, marca))

    print()
    while True:
        try:
            opcion = input("  Elige una opcion (0-18): ").strip()
            opcion = int(opcion)
            if 0 <= opcion <= len(codigos):
                if opcion == 0:
                    return None
                return codigos[opcion - 1]
        except ValueError:
            pass
        print("  Opcion no valida. Intenta de nuevo.")


def preguntar_es_musica():
    """Pregunta si el video es musical para ajustar parametros."""
    print("\n  El video contiene musica/canciones?")
    print("   [1] Si - es un video musical, anime OP/ED, concierto, etc.")
    print("   [2] No - es dialogo, entrevista, documental, etc.\n")

    while True:
        opcion = input("  Elige (1 o 2): ").strip()
        if opcion == "1":
            return True
        elif opcion == "2":
            return False
        print("  Escribe 1 o 2.")


# ── UTILIDADES ─────────────────────────────────────────
def formatear_tiempo(segundos):
    """Convierte segundos a formato SRT: HH:MM:SS,mmm"""
    h  = int(segundos // 3600)
    m  = int((segundos % 3600) // 60)
    s  = int(segundos % 60)
    ms = int(round((segundos - int(segundos)) * 1000))
    if ms >= 1000:
        ms = 999
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)


def buscar_video(carpeta_entrada):
    """Busca el primer archivo de video en la carpeta de entrada."""
    if not os.path.isdir(carpeta_entrada):
        return None
    for archivo in sorted(os.listdir(carpeta_entrada)):
        _, ext = os.path.splitext(archivo)
        if ext.lower() in EXTENSIONES_VIDEO:
            return os.path.join(carpeta_entrada, archivo)
    return None


def escribir_bloque_srt(f, contador, inicio, fin, texto):
    """Escribe un bloque SRT y hace flush inmediato al disco."""
    f.write("{}\n".format(contador))
    f.write("{} --> {}\n".format(inicio, fin))
    f.write("{}\n\n".format(texto))
    f.flush()


def convertir_a_romaji(texto, katsu):
    """Convierte texto japones (kanji/hiragana/katakana) a Romaji."""
    try:
        return katsu.romaji(texto)
    except Exception:
        return texto


def traducir_texto(texto, idioma_seg, tokenizer, modelo_nllb, dispositivo):
    """
    Traduce un segmento al espanol, configurando el idioma
    de origen correctamente segun si es japones o ingles.
    """
    # Cambiar idioma de origen segun el segmento
    codigo_origen = MAPA_IDIOMAS.get(idioma_seg, "eng_Latn")
    tokenizer.src_lang = codigo_origen

    inputs = tokenizer(
        texto, return_tensors="pt",
        truncation=True, max_length=512
    ).to(dispositivo)

    with torch.no_grad():
        tokens_trad = modelo_nllb.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(IDIOMA_DESTINO),
            max_new_tokens=256,
        )

    return tokenizer.batch_decode(tokens_trad, skip_special_tokens=True)[0]


# ── FUNCION PRINCIPAL ──────────────────────────────────
def traducir_video():
    print("=" * 60)
    print("  TRADUCTOR DE VIDEOS IA v5.2 (CODE-SWITCHING JA/EN)")
    print("=" * 60)

    # ── Rutas ──
    carpeta_script  = os.path.dirname(os.path.abspath(__file__))
    carpeta_entrada = os.path.join(carpeta_script, "1_Video_Original")
    carpeta_salida  = os.path.join(carpeta_script, "2_Subtitulos_Traducidos")
    os.makedirs(carpeta_entrada, exist_ok=True)
    os.makedirs(carpeta_salida, exist_ok=True)

    # ── Buscar video ──
    ruta_video = buscar_video(carpeta_entrada)
    if not ruta_video:
        print("\n  No se encontro ningun video en '1_Video_Original'.")
        print("  Formatos soportados: " + ", ".join(sorted(EXTENSIONES_VIDEO)))
        sys.exit(1)

    nombre_archivo = os.path.basename(ruta_video)
    nombre_base    = os.path.splitext(nombre_archivo)[0]
    print("\n  Video detectado: {}".format(nombre_archivo))

    # ── Menu interactivo ──
    idioma_manual = mostrar_menu_idioma()
    es_musica     = preguntar_es_musica()

    # Determinar si estamos en modo japones (manual o auto)
    modo_japones = (idioma_manual == "ja")

    # ── Hardware ──
    print("\n[1/4] Analizando hardware y cargando modelos...")
    if torch.cuda.is_available():
        dispositivo  = "cuda"
        tipo_computo = "float16"
        gpu_nombre   = torch.cuda.get_device_name(0)
        print("  GPU detectada: {}".format(gpu_nombre))
    else:
        dispositivo  = "cpu"
        tipo_computo = "int8"
        print("  Usando CPU (sera mas lento).")

    # ── Cargar modelos ──
    print("  Cargando Whisper ({})...".format(MODELO_WHISPER))
    modelo_whisper = WhisperModel(
        MODELO_WHISPER, device=dispositivo, compute_type=tipo_computo
    )

    print("  Cargando NLLB para traduccion...")
    tokenizer   = AutoTokenizer.from_pretrained(MODELO_NLLB)
    modelo_nllb = AutoModelForSeq2SeqLM.from_pretrained(MODELO_NLLB).to(dispositivo)

    # ── Configurar parametros de transcripcion ──
    parametros_whisper = {
        "condition_on_previous_text": False,
    }

    if idioma_manual:
        if modo_japones and es_musica:
            # ── JAPONES + MUSICA: NO forzar idioma ──
            # En vez de language="ja" (que convierte todo a katakana),
            # usamos initial_prompt en japones para GUIAR a Whisper
            # sin cerrarlo. Asi puede detectar partes en ingles.
            parametros_whisper["initial_prompt"] = PROMPT_JAPONES
            print("\n[2/4] Modo JAPONES + MUSICA (code-switching activado)")
            print("  Whisper guiado hacia japones pero abierto a ingles")
        else:
            # Otros idiomas: forzar normalmente
            parametros_whisper["language"] = idioma_manual
            print("\n[2/4] Idioma forzado: {} ({})".format(
                NOMBRES_IDIOMAS.get(idioma_manual, idioma_manual), idioma_manual
            ))
    else:
        print("\n[2/4] Escuchando audio para detectar idioma...")

    if es_musica:
        parametros_whisper["vad_filter"] = False
        parametros_whisper["beam_size"] = 5
        parametros_whisper["no_speech_threshold"] = 0.3
        parametros_whisper["word_timestamps"] = True
        print("  MODO MUSICA (VAD desactivado, sensibilidad alta)")
    else:
        parametros_whisper["vad_filter"] = True
        parametros_whisper["vad_parameters"] = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
        }
        print("  MODO DIALOGO (VAD activado)")

    # ── Transcribir ──
    segmentos_gen, info = modelo_whisper.transcribe(ruta_video, **parametros_whisper)

    idioma_detectado = idioma_manual if idioma_manual else info.language

    if not idioma_manual:
        probabilidad = info.language_probability
        print("  Idioma detectado: {} (confianza: {:.0%})".format(
            idioma_detectado, probabilidad
        ))
        if probabilidad < 0.6:
            print("  AVISO: Confianza baja. Prueba eligiendo el idioma manualmente.")

    # Para el flujo general: si eligio japones o se detecto japones
    es_japones = (idioma_detectado == "ja")

    if es_japones and not ROMAJI_DISPONIBLE:
        print("\n  Japones detectado pero 'cutlet' no esta instalado.")
        print("  Instalalo con:  pip install cutlet unidic-lite")
        print("  Se generara solo la traduccion al espanol.\n")

    # ── Preparar Romaji si aplica ──
    katsu = None
    if es_japones and ROMAJI_DISPONIBLE:
        print("  Inicializando motor de Romaji (cutlet + MeCab)...")
        katsu = cutlet.Cutlet()
        katsu.use_foreign_spelling = False

    # ── Rutas de salida ──
    ruta_srt_es     = os.path.join(carpeta_salida, "{}_ES.srt".format(nombre_base))
    ruta_srt_romaji = os.path.join(carpeta_salida, "{}_ROMAJI.srt".format(nombre_base))
    ruta_srt_jpn    = None
    if es_japones:
        ruta_srt_jpn = os.path.join(carpeta_salida, "{}_JA.srt".format(nombre_base))

    print("\n[3/4] Transcribiendo y traduciendo...\n")

    if es_japones and es_musica:
        print("  >> Code-switching activo: cada linea se analiza para")
        print("     detectar si es japones o ingles automaticamente.\n")

    print("       {:<28} {}".format("TIEMPO", "TEXTO"))
    print("       {} {}".format("-" * 28, "-" * 44))

    # ── Procesar segmentos ──
    contador         = 1
    errores          = 0
    seg_ja_count     = 0
    seg_en_count     = 0
    segmentos_romaji = []
    segmentos_jpn    = []

    with open(ruta_srt_es, "w", encoding="utf-8") as f_es:
        for segmento in segmentos_gen:
            texto_original = segmento.text.strip()

            # Saltar segmentos vacios o solo puntuacion
            if not texto_original:
                continue
            caracteres_basura = set(".,!?-...。、！？ー　 \t\n\"'()[]")
            if all(c in caracteres_basura for c in texto_original):
                continue

            inicio = formatear_tiempo(segmento.start)
            fin    = formatear_tiempo(segmento.end)

            try:
                # ── Detectar idioma del segmento ──
                if es_japones and es_musica:
                    idioma_seg = detectar_idioma_texto(texto_original)
                elif es_japones:
                    idioma_seg = "ja"
                else:
                    idioma_seg = idioma_detectado

                # Contadores
                if idioma_seg == "ja":
                    seg_ja_count += 1
                else:
                    seg_en_count += 1

                # ── Traducir al espanol (con idioma correcto) ──
                texto_es = traducir_texto(
                    texto_original, idioma_seg,
                    tokenizer, modelo_nllb, dispositivo
                )

                # ── Escribir SRT espanol ──
                escribir_bloque_srt(f_es, contador, inicio, fin, texto_es)

                # ── Guardar original japones + Romaji ──
                if es_japones:
                    segmentos_jpn.append((contador, inicio, fin, texto_original))

                    if katsu:
                        if idioma_seg == "ja":
                            # Segmento japones: romanizar
                            texto_romaji = convertir_a_romaji(texto_original, katsu)
                        else:
                            # Segmento ingles: mantener como esta
                            texto_romaji = texto_original
                        segmentos_romaji.append((contador, inicio, fin, texto_romaji))

                # ── Mostrar progreso ──
                etiqueta = "[JA]" if idioma_seg == "ja" else "[EN]"

                if es_japones:
                    print("  [{} -> {}] {}".format(inicio, fin, etiqueta))
                    print("     Original: {}".format(texto_original))
                    if katsu and idioma_seg == "ja":
                        print("     Romaji:   {}".format(
                            convertir_a_romaji(texto_original, katsu)
                        ))
                    print("     Espanol:  {}".format(texto_es))
                    print()
                else:
                    print("  [{} -> {}]  {}".format(inicio, fin, texto_es))

                contador += 1

            except Exception as e:
                errores += 1
                print("  Error en segmento [{}]: {}".format(inicio, e))
                continue

    # ── Escribir SRT japones original ──
    if segmentos_jpn and ruta_srt_jpn:
        with open(ruta_srt_jpn, "w", encoding="utf-8") as f_jpn:
            for num, ini, fi, txt in segmentos_jpn:
                escribir_bloque_srt(f_jpn, num, ini, fi, txt)

    # ── Escribir SRT Romaji ──
    if segmentos_romaji:
        with open(ruta_srt_romaji, "w", encoding="utf-8") as f_rom:
            for num, ini, fi, txt in segmentos_romaji:
                escribir_bloque_srt(f_rom, num, ini, fi, txt)

    # ── Resumen final ──
    print("=" * 60)
    print("  [4/4] PROCESO TERMINADO!")
    print("=" * 60)

    total = contador - 1
    print("\n  Archivos generados en '2_Subtitulos_Traducidos':")
    print("     {}  (Espanol)".format(os.path.basename(ruta_srt_es)))

    if segmentos_jpn and ruta_srt_jpn:
        print("     {}  (Japones original)".format(os.path.basename(ruta_srt_jpn)))

    if segmentos_romaji:
        print("     {}  (Romaji)".format(os.path.basename(ruta_srt_romaji)))

    if es_japones and es_musica and total > 0:
        print("\n  Code-switching detectado:")
        print("     {} lineas en JAPONES".format(seg_ja_count))
        print("     {} lineas en INGLES".format(seg_en_count))
        print("     (cada linea se tradujo desde su idioma correcto)")

    if segmentos_romaji:
        print("\n  El archivo Romaji contiene:")
        print("     - Lineas japonesas -> transliteradas a alfabeto latino")
        print("     - Lineas inglesas  -> texto original (ya es latino)")

    if errores > 0:
        print("\n  {} segmento(s) no pudieron procesarse.".format(errores))

    print("\n  {} subtitulos generados correctamente.".format(total))
    print()


if __name__ == "__main__":
    traducir_video()
