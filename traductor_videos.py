import os
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

def traducir_video():
    print("=== TRADUCTOR DE VIDEOS IA (LOCAL) ===\n")
    
    ruta_video = input("Arrastra tu video aquí o escribe la ruta del archivo: ").strip().strip('"')
    
    idioma_origen = input("¿En qué idioma está tu video? (ej: japones, ingles, español): ").strip().lower()
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
    
    # --- DETECCIÓN DE HARDWARE ---
    if torch.cuda.is_available():
        dispositivo = "cuda"
        tipo_computo = "float16"
        print("✅ ¡Tarjeta Gráfica NVIDIA (GPU) detectada! Activando aceleración a máxima velocidad.")
    else:
        dispositivo = "cpu"
        tipo_computo = "int8"
        print("💻 Usando el Procesador (CPU) en modo optimizado de bajo consumo.")
    # -----------------------------

    # Cargamos Whisper
    modelo_whisper = WhisperModel("base", device=dispositivo, compute_type=tipo_computo)
    
    # Cargamos NLLB-200 y lo enviamos al hardware detectado
    tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    modelo_nllb = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(dispositivo)

    print(f"\n[2/3] Extrayendo audio y transcribiendo desde el {idioma_origen}...")
    segmentos, _ = modelo_whisper.transcribe(ruta_video, language=codigos_origen["whisper"])

    nombre_srt = f"subtitulos_{idioma_destino}.srt"
    
    print(f"\n[3/3] Traduciendo al {idioma_destino} y creando archivo SRT...\n")
    with open(nombre_srt, "w", encoding="utf-8") as archivo_srt:
        contador = 1
        for segmento in segmentos:
            texto_original = segmento.text
            
            # Preparamos el texto y lo enviamos al hardware correcto
            inputs = tokenizer(texto_original, return_tensors="pt").to(dispositivo)
            
            # Generamos la traducción
            translated_tokens = modelo_nllb.generate(
                **inputs, 
                forced_bos_token_id=tokenizer.lang_code_to_id[codigos_destino["nllb"]]
            )
            texto_traducido = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
            
            # Formateamos tiempos y guardamos
            inicio = formatear_tiempo(segmento.start)
            fin = formatear_tiempo(segmento.end)
            
            archivo_srt.write(f"{contador}\n")
            archivo_srt.write(f"{inicio} --> {fin}\n")
            archivo_srt.write(f"{texto_traducido}\n\n")
            
            print(f"[{inicio}] {texto_traducido}")
            contador += 1

    print(f"\n¡PROCESO TERMINADO! Tu traducción se guardó en: {nombre_srt}")

if __name__ == "__main__":
    traducir_video()