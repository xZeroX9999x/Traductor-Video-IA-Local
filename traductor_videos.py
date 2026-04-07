import os
import sys
import glob
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# ── Intentar importar cutlet (Romaji) ──────────────────
try:
    import cutlet
    ROMAJI_DISPONIBLE = True
except ImportError:
    ROMAJI_DISPONIBLE = False

# ── CONFIGURACIÓN ──────────────────────────────────────
MODELO_WHISPER = "medium"          # "small" | "medium" | "large-v3"
IDIOMA_DESTINO = "spa_Latn"        # Código NLLB del idioma destino
MODELO_NLLB    = "facebook/nllb-200-distilled-600M"

# Mapa Whisper → NLLB
MAPA_IDIOMAS = {
    "en": "eng_Latn", "ja": "jpn_Jpan", "fr": "fra_Latn",
    "de": "deu_Latn", "it": "ita_Latn", "pt": "por_Latn",
    "ru": "rus_Cyrl", "zh": "zho_Hans", "ko": "kor_Hang",
    "nl": "nld_Latn", "tr": "tur_Latn", "ar": "arb_Arab",
    "sv": "swe_Latn", "pl": "pol_Latn", "hi": "hin_Deva",
    "es": "spa_Latn", "id": "ind_Latn", "vi": "vie_Latn",
}

EXTENSIONES_VIDEO = {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.flv', '.wmv'}


# ── UTILIDADES ─────────────────────────────────────────
def formatear_tiempo(segundos: float) -> str:
    """Convierte segundos a formato SRT: HH:MM:SS,mmm"""
    h  = int(segundos // 3600)
    m  = int((segundos % 3600) // 60)
    s  = int(segundos % 60)
    ms = int(round((segundos - int(segundos)) * 1000))
    if ms >= 1000:          # protección contra redondeo
        ms = 999
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def buscar_video(carpeta_entrada: str):
    """Busca el primer archivo de video en la carpeta de entrada."""
    if not os.path.isdir(carpeta_entrada):
        return None

    for archivo in sorted(os.listdir(carpeta_entrada)):
        _, ext = os.path.splitext(archivo)
        if ext.lower() in EXTENSIONES_VIDEO:
            return os.path.join(carpeta_entrada, archivo)
    return None


def escribir_bloque_srt(f, contador: int, inicio: str, fin: str, texto: str):
    """Escribe un bloque SRT y hace flush inmediato al disco."""
    f.write(f"{contador}\n")
    f.write(f"{inicio} --> {fin}\n")
    f.write(f"{texto}\n\n")
    f.flush()


def convertir_a_romaji(texto: str, katsu) -> str:
    """
    Convierte texto japonés (kanji/hiragana/katakana) a Romaji.
    
    Usa la librería cutlet que internamente emplea MeCab + UniDic
    para segmentar el texto japonés y asignar lecturas correctas.
    
    Ejemplo:
        今日はとても暑いです → kyou wa totemo atsui desu
        残酷な天使のテーゼ  → zankoku na tenshi no tēze
    """
    try:
        return katsu.romaji(texto)
    except Exception:
        # Fallback: devolver el texto original si falla
        return texto


# ── FUNCIÓN PRINCIPAL ──────────────────────────────────
def traducir_video():
    print("=" * 55)
    print("  TRADUCTOR DE VIDEOS IA — V5 (CON ROMAJI JAPONÉS)")
    print("=" * 55)

    # ── Rutas ──
    carpeta_script  = os.path.dirname(os.path.abspath(__file__))
    carpeta_entrada = os.path.join(carpeta_script, "1_Video_Original")
    carpeta_salida  = os.path.join(carpeta_script, "2_Subtitulos_Traducidos")
    os.makedirs(carpeta_entrada, exist_ok=True)
    os.makedirs(carpeta_salida, exist_ok=True)

    # ── Buscar video ──
    ruta_video = buscar_video(carpeta_entrada)
    if not ruta_video:
        print("\n❌ Error: No se encontró ningún video en '1_Video_Original'.")
        print("   Formatos soportados:", ", ".join(sorted(EXTENSIONES_VIDEO)))
        sys.exit(1)

    nombre_archivo = os.path.basename(ruta_video)
    nombre_base    = os.path.splitext(nombre_archivo)[0]
    print(f"\n🎬 Video detectado: {nombre_archivo}")

    # ── Hardware ──
    print("\n[1/4] Analizando hardware y cargando modelos...")
    if torch.cuda.is_available():
        dispositivo  = "cuda"
        tipo_computo = "float16"
        gpu_nombre   = torch.cuda.get_device_name(0)
        print(f"  ✅ GPU detectada: {gpu_nombre}")
    else:
        dispositivo  = "cpu"
        tipo_computo = "int8"
        print("  💻 Usando CPU (será más lento).")

    # ── Cargar modelos ──
    print(f"  ⏳ Cargando Whisper ({MODELO_WHISPER})...")
    modelo_whisper = WhisperModel(
        MODELO_WHISPER, device=dispositivo, compute_type=tipo_computo
    )

    print(f"  ⏳ Cargando NLLB para traducción...")
    tokenizer  = AutoTokenizer.from_pretrained(MODELO_NLLB)
    modelo_nllb = AutoModelForSeq2SeqLM.from_pretrained(MODELO_NLLB).to(dispositivo)

    # ── Transcripción + detección de idioma ──
    print("\n[2/4] Escuchando audio para detectar idioma...")
    segmentos_gen, info = modelo_whisper.transcribe(
        ruta_video,
        condition_on_previous_text=False,
        vad_filter=True,              # filtra silencios y ruido
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=300,
        ),
    )
    idioma_detectado = info.language
    probabilidad     = info.language_probability

    print(f"  🌍 Idioma: {idioma_detectado} (confianza: {probabilidad:.0%})")

    es_japones = (idioma_detectado == "ja")

    if es_japones and not ROMAJI_DISPONIBLE:
        print("\n  ⚠️  Japonés detectado pero 'cutlet' no está instalado.")
        print("     Instálalo con:  pip install cutlet unidic-lite")
        print("     Se generará solo la traducción al español.\n")

    # ── Preparar Romaji si aplica ──
    katsu = None
    if es_japones and ROMAJI_DISPONIBLE:
        print("  🇯🇵 Inicializando motor de Romaji (cutlet + MeCab)...")
        katsu = cutlet.Cutlet()
        katsu.use_foreign_spelling = False  # evita mezclar inglés

    # ── Configurar traducción ──
    codigo_origen  = MAPA_IDIOMAS.get(idioma_detectado, "eng_Latn")
    tokenizer.src_lang = codigo_origen

    # ── Rutas de salida ──
    ruta_srt_es     = os.path.join(carpeta_salida, f"{nombre_base}_ES.srt")
    ruta_srt_romaji = os.path.join(carpeta_salida, f"{nombre_base}_ROMAJI.srt")

    print(f"\n[3/4] Transcribiendo y traduciendo...\n")
    print(f"       {'TIEMPO':<28} {'ESPAÑOL'}")
    print(f"       {'-'*28} {'-'*40}")

    # ── Procesar segmentos ──
    contador    = 1
    errores     = 0
    segmentos_romaji = []

    with open(ruta_srt_es, "w", encoding="utf-8") as f_es:
        for segmento in segmentos_gen:
            texto_original = segmento.text.strip()

            # Saltar segmentos vacíos o solo puntuación
            if not texto_original or all(c in ".,!?-…。、！？―　 " for c in texto_original):
                continue

            inicio = formatear_tiempo(segmento.start)
            fin    = formatear_tiempo(segmento.end)

            try:
                # ── Traducir a español ──
                inputs = tokenizer(
                    texto_original, return_tensors="pt",
                    truncation=True, max_length=512
                ).to(dispositivo)

                with torch.no_grad():
                    tokens_trad = modelo_nllb.generate(
                        **inputs,
                        forced_bos_token_id=tokenizer.convert_tokens_to_ids(IDIOMA_DESTINO),
                        max_new_tokens=256,
                    )

                texto_es = tokenizer.batch_decode(
                    tokens_trad, skip_special_tokens=True
                )[0]

                # ── Escribir SRT español ──
                escribir_bloque_srt(f_es, contador, inicio, fin, texto_es)

                # ── Romaji (si es japonés) ──
                if es_japones and katsu:
                    texto_romaji = convertir_a_romaji(texto_original, katsu)
                    segmentos_romaji.append((contador, inicio, fin, texto_romaji))

                print(f"  [{inicio} → {fin}]  {texto_es}")
                contador += 1

            except Exception as e:
                errores += 1
                print(f"  ⚠️  Error en segmento [{inicio}]: {e}")
                continue

    # ── Escribir SRT Romaji ──
    if segmentos_romaji:
        with open(ruta_srt_romaji, "w", encoding="utf-8") as f_rom:
            for num, ini, fi, txt in segmentos_romaji:
                escribir_bloque_srt(f_rom, num, ini, fi, txt)

    # ── Resumen final ──
    print(f"\n{'=' * 55}")
    print(f"  [4/4] ¡PROCESO TERMINADO!")
    print(f"{'=' * 55}")
    print(f"\n  📁 Archivos generados en '2_Subtitulos_Traducidos':")
    print(f"     • {os.path.basename(ruta_srt_es)}  (Español)")

    if segmentos_romaji:
        print(f"     • {os.path.basename(ruta_srt_romaji)}  (Romaji japonés)")
        print(f"\n  💡 El archivo Romaji contiene la letra transliterada")
        print(f"     al alfabeto latino para que puedas seguir la canción.")

    if errores > 0:
        print(f"\n  ⚠️  {errores} segmento(s) no pudieron procesarse.")

    total = contador - 1
    print(f"\n  ✅ {total} subtítulos generados correctamente.")
    print()


if __name__ == "__main__":
    traducir_video()
