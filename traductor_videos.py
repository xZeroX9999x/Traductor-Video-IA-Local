import os
import sys
import subprocess
import shutil
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
IDIOMA_DESTINO = "spa_Latn"
MODELO_NLLB    = "facebook/nllb-200-distilled-600M"

MAPA_IDIOMAS = {
    "en": "eng_Latn", "ja": "jpn_Jpan", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ko": "kor_Hang",
    "nl": "nld_Latn", "tr": "tur_Latn", "ar": "arb_Arab",
    "sv": "swe_Latn", "pl": "pol_Latn", "hi": "hin_Deva",
    "es": "spa_Latn", "id": "ind_Latn", "vi": "vie_Latn",
}

NOMBRES_IDIOMAS = {
    "en": "Ingles",    "ja": "Japones",    "fr": "Frances",
    "de": "Aleman",    "it": "Italiano",   "pt": "Portugues",
    "ru": "Ruso",      "zh": "Chino",      "ko": "Coreano",
    "nl": "Holandes",  "tr": "Turco",      "ar": "Arabe",
    "sv": "Sueco",     "pl": "Polaco",     "hi": "Hindi",
    "es": "Espanol",   "id": "Indonesio",  "vi": "Vietnamita",
}

EXTENSIONES_VIDEO = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}

PROMPT_JAPONES = (
    "音楽、歌詞、日本語と英語が混ざった歌。"
    "ロック、メタル、ビジュアル系。"
)


# ── SEPARACION VOCAL CON DEMUCS ────────────────────────
def verificar_demucs():
    """Verifica si Demucs esta instalado."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "demucs", "--help"],
            capture_output=True, timeout=10
        )
        return True
    except Exception:
        return False


def extraer_audio(ruta_video, ruta_audio):
    """Extrae el audio del video usando FFmpeg."""
    print("     Extrayendo audio del video...")
    cmd = [
        "ffmpeg", "-i", ruta_video,
        "-vn",                    # sin video
        "-acodec", "pcm_s16le",   # WAV sin compresion
        "-ar", "44100",           # 44.1kHz (calidad CD)
        "-ac", "2",               # estereo
        "-y",                     # sobreescribir
        ruta_audio
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        print("     Error FFmpeg: {}".format(result.stderr.decode("utf-8", errors="ignore")[-200:]))
        return False
    return True


def separar_vocales(ruta_audio, carpeta_temp):
    """
    Usa Demucs para separar la voz de los instrumentos.

    Demucs (de Meta/Facebook) es una red neuronal entrenada
    para separar stems de audio. Con --two-stems=vocals
    genera dos archivos:
      - vocals.wav    (solo la voz del cantante)
      - no_vocals.wav (solo instrumentos)

    Esto elimina guitarras, bateria, bajo, etc. y le da
    a Whisper audio limpio para transcribir.
    """
    print("     Separando voz de instrumentos con Demucs IA...")
    print("     (esto puede tardar unos minutos)")

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",     # solo separar voz vs resto
        "-n", "htdemucs",         # modelo mas preciso
        "--out", carpeta_temp,    # carpeta de salida
        ruta_audio
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("     Error Demucs: {}".format(result.stderr[-300:] if result.stderr else "desconocido"))
        return None

    # Demucs guarda en: carpeta_temp/htdemucs/nombre_archivo/vocals.wav
    nombre_audio = os.path.splitext(os.path.basename(ruta_audio))[0]
    ruta_vocales = os.path.join(carpeta_temp, "htdemucs", nombre_audio, "vocals.wav")

    if os.path.exists(ruta_vocales):
        tamano_mb = os.path.getsize(ruta_vocales) / (1024 * 1024)
        print("     Voz aislada correctamente ({:.1f} MB)".format(tamano_mb))
        return ruta_vocales
    else:
        print("     No se encontro el archivo de vocales.")
        return None


# ── DETECCION DE IDIOMA POR SEGMENTO ──────────────────
def detectar_idioma_texto(texto):
    """
    Analiza caracteres Unicode para determinar si un
    segmento es japones o ingles.

    Rangos Unicode japoneses:
      Hiragana:  U+3040 - U+309F  (あいうえお)
      Katakana:  U+30A0 - U+30FF  (アイウエオ)
      Kanji:     U+4E00 - U+9FFF  (漢字)
      CJK Ext A: U+3400 - U+4DBF
      Simbolos:  U+3000 - U+303F  (。、「」)
      Kata ext:  U+31F0 - U+31FF

    Si >15% de caracteres (sin espacios) son japoneses -> "ja"
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
        if (0x3040 <= cp <= 0x309F or    # Hiragana
            0x30A0 <= cp <= 0x30FF or    # Katakana
            0x4E00 <= cp <= 0x9FFF or    # Kanji
            0x3400 <= cp <= 0x4DBF or    # CJK Extension A
            0x3000 <= cp <= 0x303F or    # Simbolos CJK
            0x31F0 <= cp <= 0x31FF):     # Katakana extendido
            japones += 1

    if total == 0:
        return "en"

    return "ja" if (japones / total) > 0.15 else "en"


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
    """Pregunta si el video es musical."""
    print("\n  El video contiene musica/canciones?")
    print("   [1] Si - video musical, anime OP/ED, concierto, etc.")
    print("   [2] No - dialogo, entrevista, documental, etc.\n")

    while True:
        opcion = input("  Elige (1 o 2): ").strip()
        if opcion == "1":
            return True
        elif opcion == "2":
            return False
        print("  Escribe 1 o 2.")


# ── UTILIDADES ─────────────────────────────────────────
def formatear_tiempo(segundos):
    h  = int(segundos // 3600)
    m  = int((segundos % 3600) // 60)
    s  = int(segundos % 60)
    ms = int(round((segundos - int(segundos)) * 1000))
    if ms >= 1000:
        ms = 999
    return "{:02d}:{:02d}:{:02d},{:03d}".format(h, m, s, ms)


def buscar_video(carpeta_entrada):
    if not os.path.isdir(carpeta_entrada):
        return None
    for archivo in sorted(os.listdir(carpeta_entrada)):
        _, ext = os.path.splitext(archivo)
        if ext.lower() in EXTENSIONES_VIDEO:
            return os.path.join(carpeta_entrada, archivo)
    return None


def escribir_bloque_srt(f, contador, inicio, fin, texto):
    f.write("{}\n".format(contador))
    f.write("{} --> {}\n".format(inicio, fin))
    f.write("{}\n\n".format(texto))
    f.flush()


def convertir_a_romaji(texto, katsu):
    try:
        return katsu.romaji(texto)
    except Exception:
        return texto


def traducir_texto(texto, idioma_seg, tokenizer, modelo_nllb, dispositivo):
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
    print("  TRADUCTOR DE VIDEOS IA v6 (SEPARACION VOCAL + ROMAJI)")
    print("=" * 60)

    # ── Rutas ──
    carpeta_script  = os.path.dirname(os.path.abspath(__file__))
    carpeta_entrada = os.path.join(carpeta_script, "1_Video_Original")
    carpeta_salida  = os.path.join(carpeta_script, "2_Subtitulos_Traducidos")
    carpeta_temp    = os.path.join(carpeta_script, "3_Temp")
    os.makedirs(carpeta_entrada, exist_ok=True)
    os.makedirs(carpeta_salida, exist_ok=True)
    os.makedirs(carpeta_temp, exist_ok=True)

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
    modo_japones  = (idioma_manual == "ja")

    # ── Hardware ──
    pasos = "5" if es_musica else "4"
    print("\n[1/{}] Analizando hardware y cargando modelos...".format(pasos))

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

    # ── SEPARACION VOCAL (solo para musica) ────────────
    ruta_audio_para_whisper = ruta_video  # por defecto: audio del video
    demucs_ok = False

    if es_musica:
        print("\n[2/{}] Separando voz de instrumentos...".format(pasos))

        tiene_demucs = verificar_demucs()

        if not tiene_demucs:
            print("  Demucs no esta instalado.")
            print("  Para MEJOR precision en musica, instala con:")
            print("    pip install demucs")
            print("  Continuando sin separacion vocal...\n")
        else:
            # Paso A: Extraer audio del video a WAV
            ruta_audio_wav = os.path.join(carpeta_temp, "audio_original.wav")
            exito_audio = extraer_audio(ruta_video, ruta_audio_wav)

            if exito_audio:
                # Paso B: Separar vocales con Demucs
                ruta_vocales = separar_vocales(ruta_audio_wav, carpeta_temp)

                if ruta_vocales:
                    ruta_audio_para_whisper = ruta_vocales
                    demucs_ok = True
                    print("     Whisper usara la pista VOCAL aislada.")
                else:
                    print("     Fallo la separacion. Usando audio original.")
            else:
                print("     Fallo la extraccion de audio. Usando video original.")

    # ── Configurar Whisper ──
    paso_idioma = "3" if es_musica else "2"
    parametros_whisper = {
        "condition_on_previous_text": False,
    }

    if idioma_manual:
        if modo_japones and es_musica:
            parametros_whisper["initial_prompt"] = PROMPT_JAPONES
            print("\n[{}/{}] Modo JAPONES + MUSICA (code-switching activado)".format(
                paso_idioma, pasos
            ))
        else:
            parametros_whisper["language"] = idioma_manual
            print("\n[{}/{}] Idioma forzado: {} ({})".format(
                paso_idioma, pasos,
                NOMBRES_IDIOMAS.get(idioma_manual, idioma_manual), idioma_manual
            ))
    else:
        print("\n[{}/{}] Escuchando audio para detectar idioma...".format(
            paso_idioma, pasos
        ))

    if es_musica:
        parametros_whisper["vad_filter"] = False
        parametros_whisper["beam_size"] = 5
        parametros_whisper["no_speech_threshold"] = 0.3
        parametros_whisper["word_timestamps"] = True
        if demucs_ok:
            print("  MODO MUSICA + VOZ AISLADA (maxima precision)")
        else:
            print("  MODO MUSICA sin separacion vocal")
    else:
        parametros_whisper["vad_filter"] = True
        parametros_whisper["vad_parameters"] = {
            "min_silence_duration_ms": 500,
            "speech_pad_ms": 300,
        }
        print("  MODO DIALOGO (VAD activado)")

    # ── Transcribir ──
    segmentos_gen, info = modelo_whisper.transcribe(
        ruta_audio_para_whisper, **parametros_whisper
    )

    idioma_detectado = idioma_manual if idioma_manual else info.language

    if not idioma_manual:
        probabilidad = info.language_probability
        print("  Idioma detectado: {} (confianza: {:.0%})".format(
            idioma_detectado, probabilidad
        ))
        if probabilidad < 0.6:
            print("  AVISO: Confianza baja. Prueba eligiendo el idioma manualmente.")

    es_japones = (idioma_detectado == "ja")

    if es_japones and not ROMAJI_DISPONIBLE:
        print("\n  Japones detectado pero 'cutlet' no esta instalado.")
        print("  Instalalo con:  pip install cutlet unidic-lite\n")

    # ── Preparar Romaji ──
    katsu = None
    if es_japones and ROMAJI_DISPONIBLE:
        print("  Inicializando Romaji (cutlet + MeCab)...")
        katsu = cutlet.Cutlet()
        katsu.use_foreign_spelling = False

    # ── Rutas de salida ──
    ruta_srt_es     = os.path.join(carpeta_salida, "{}_ES.srt".format(nombre_base))
    ruta_srt_romaji = os.path.join(carpeta_salida, "{}_ROMAJI.srt".format(nombre_base))
    ruta_srt_jpn    = None
    if es_japones:
        ruta_srt_jpn = os.path.join(carpeta_salida, "{}_JA.srt".format(nombre_base))

    paso_transcribir = "4" if es_musica else "3"
    print("\n[{}/{}] Transcribiendo y traduciendo...\n".format(paso_transcribir, pasos))

    if es_japones and es_musica:
        print("  >> Code-switching activo: JA/EN por segmento\n")

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

                if idioma_seg == "ja":
                    seg_ja_count += 1
                else:
                    seg_en_count += 1

                # ── Traducir ──
                texto_es = traducir_texto(
                    texto_original, idioma_seg,
                    tokenizer, modelo_nllb, dispositivo
                )

                escribir_bloque_srt(f_es, contador, inicio, fin, texto_es)

                # ── Japones: original + Romaji ──
                if es_japones:
                    segmentos_jpn.append((contador, inicio, fin, texto_original))
                    if katsu:
                        if idioma_seg == "ja":
                            texto_romaji = convertir_a_romaji(texto_original, katsu)
                        else:
                            texto_romaji = texto_original
                        segmentos_romaji.append((contador, inicio, fin, texto_romaji))

                # ── Progreso ──
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

    # ── Limpiar archivos temporales ──
    if os.path.isdir(carpeta_temp):
        try:
            shutil.rmtree(carpeta_temp)
            print("\n  Archivos temporales eliminados.")
        except Exception:
            print("\n  Nota: No se pudieron eliminar los archivos temporales en '3_Temp'.")

    # ── Resumen final ──
    paso_final = pasos
    print("\n" + "=" * 60)
    print("  [{}/{}] PROCESO TERMINADO!".format(paso_final, pasos))
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

    if demucs_ok:
        print("\n  Demucs aislo la voz correctamente.")
        print("  La transcripcion fue sobre audio vocal limpio.")

    if segmentos_romaji:
        print("\n  El archivo Romaji contiene:")
        print("     Lineas JA -> transliteradas a alfabeto latino")
        print("     Lineas EN -> texto original (ya es latino)")

    if errores > 0:
        print("\n  {} segmento(s) con errores.".format(errores))

    print("\n  {} subtitulos generados.".format(total))
    print()


if __name__ == "__main__":
    traducir_video()
