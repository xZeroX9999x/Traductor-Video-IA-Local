import os
import shutil
import glob
import torch
from faster_whisper import WhisperModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# 1. DICCIONARIO DE IDIOMAS
idiomas_soportados = {
    "español": {"whisper": "es", "nllb": "spa_Latn"},
    "ingles": {"whisper": "en", "nllb": "eng_Latn"},
    "japones": {"whisper": "ja", "nllb": "jpn_Jpan"},
    "frances": {"whisper": "fr", "nllb": "fra_Latn"},
    "aleman": {"whisper": "de", "nllb": "deu_Latn"}
}

def formatear_tiempo(segundos):
    """Convierte segundos al formato de subtítulos SRT (HH:MM:SS,mmm)"""
    horas = int(segundos // 3600)
    minutos = int((segundos % 3600) // 60)
    segs = int(segundos % 60)
    milisegundos = int((segundos - int(segundos)) * 1000)
    return f"{horas:02d}:{minutos:02d}:{segs:02d},{milisegundos:03d}"

def buscar_video():
    """Busca automáticamente el primer archivo de video en la carpeta actual."""
    extensiones = ['*.mp4', '*.mkv', '*.avi', '*.mov']
    videos_encontrados = []
    for ext in extensiones:
        videos_encontrados.extend(glob.glob(ext))
    
    if not videos_encontrados:
        return None
    return videos_encontrados[0] # Toma el primer video que encuentre

def traducir_video():
    print("=== TRADUCTOR DE VIDEOS IA (LOCAL) ===\n")
    
    # --- NUEVA LÓGICA DE AUTODETECCIÓN ---
    ruta_video = buscar_video()
    
    if not ruta_video:
        print("❌ Error: No se encontró ningún video (.mp4, .mkv, .avi) en esta carpeta.")
        print("Por favor, pon tu archivo de video junto a este script y vuelve a intentarlo.")
        return

    print(f"🎬 Video detectado automáticamente: {ruta_video}")
    
    idioma_origen = input("\n¿En qué idioma está tu video? (ej: japones, ingles, español): ").strip().lower()
    if idioma_origen not in idiomas_soportados:
        print("Idioma de origen no configurado en el diccionario.")
        return

    idioma_destino = input("¿A qué idioma quieres traducirlo? (ej: español, ingles): ").strip().lower()
    if idioma_destino not in idiomas_soportados:
        print("Idioma de destino no configurado en el diccionario.")
        return

    codigos_origen = idiomas_soportados[idioma_origen]
    codigos_destino = idiomas_soportados[idioma_destino]

    print("\n[1/3] Analizando hardware y cargando modelos...")
    
    if torch.cuda.is_available():
        dispositivo = "cuda"
        tipo_computo = "float16"
        print("✅ ¡Tarjeta Gráfica NVIDIA (GPU) detectada! Activando aceleración a máxima velocidad.")
    else:
        dispositivo = "cpu"
        tipo_computo = "int8"
        print("💻 Usando el Procesador (CPU) en modo optimizado de bajo consumo.")

    modelo_whisper = WhisperModel("base", device=dispositivo, compute_type=tipo_computo)
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    modelo_nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(dispositivo)

    # --- CREACIÓN DE CARPETAS ---
    carpeta_video = "1_Video_Original"
    carpeta_subs = "2_Subtitulos_Traducidos"
    os.makedirs(carpeta_video, exist_ok=True)
    os.makedirs(carpeta_subs, exist_ok=True)

    print(f"\n[2/3] Extrayendo audio y transcribiendo desde el {idioma_origen}...")
    segmentos, _ = modelo_whisper.transcribe(ruta_video, language=codigos_origen["whisper"])

    nombre_srt = f"subtitulos_{idioma_destino}.srt"
    ruta_srt = os.path.join(carpeta_subs, nombre_srt) # Guardar dentro de la nueva carpeta
    
    print(f"\n[3/3] Traduciendo al {idioma_destino} y creando archivo SRT...\n")
    
    with open(ruta_srt, "w", encoding="utf-8") as archivo_srt:
        contador = 1
        for segmento in segmentos:
            texto_original = segmento.text
            inputs = tokenizer(texto_original, return_tensors="pt").to(dispositivo)
            
            translated_tokens = modelo_nllb.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.lang_code_to_id[codigos_destino["nllb"]]
            )
            texto_traducido = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            inicio = formatear_tiempo(segmento.start)
            fin = formatear_tiempo(segmento.end)
            
            archivo_srt.write(f"{contador}\n")
            archivo_srt.write(f"{inicio} --> {fin}\n")
            archivo_srt.write(f"{texto_traducido}\n\n")
            
            print(f"[{inicio}] {texto_traducido}")
            contador += 1

    # Mover el video original a su nueva carpeta al terminar todo
    nueva_ruta_video = os.path.join(carpeta_video, ruta_video)
    shutil.move(ruta_video, nueva_ruta_video)

    print(f"\n¡PROCESO TERMINADO CON ÉXITO!")
    print(f"📁 Tu video se guardó de forma segura en la carpeta: '{carpeta_video}'")
    print(f"📁 Tus subtítulos están listos en la carpeta: '{carpeta_subs}'")

if __name__ == "__main__":
    traducir_video()
