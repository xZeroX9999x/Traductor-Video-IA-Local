# 🎬 AI Local Video Translator (Multi-Language)

Un script en Python que utiliza Inteligencia Artificial de código abierto para transcribir y traducir videos localmente, sin límites de tiempo, sin cuotas de API y garantizando total privacidad.

Este proyecto combina el reconocimiento de voz de **Faster-Whisper** con la potencia de traducción de **NLLB-200** (No Language Left Behind de Meta) para generar archivos de subtítulos (`.srt`) precisos y sincronizados.

## ✨ Características Principales

* **100% Local y Gratuito:** No depende de APIs comerciales (como Google o DeepL). Sin límites de caracteres ni suscripciones.
* **Soporte Multi-Idioma:** Traduce entre más de 200 idiomas soportados por el modelo NLLB.
* **Detección de Hardware Inteligente:** Detecta automáticamente si existe una GPU NVIDIA disponible a través de CUDA para acelerar el procesamiento, o utiliza la CPU de forma optimizada si no hay gráfica dedicada.
* **Sincronización Automática:** Extrae el audio, lo transcribe y genera subtítulos `.srt` con marcas de tiempo exactas.

## 🛠️ Requisitos Previos

Antes de ejecutar el script, asegúrate de tener instalado lo siguiente en tu sistema:

1. **Python 3.8 o superior** (Asegúrate de marcar "Add to PATH" durante la instalación).
2. **FFmpeg**: Herramienta necesaria para procesar el audio del video.
3. **NVIDIA CUDA Toolkit (Opcional pero recomendado):** Si tienes una tarjeta gráfica NVIDIA, instala CUDA Toolkit 12.1 para acelerar drásticamente el proceso.

## 🚀 Instalación Rápida

Instala las dependencias de Inteligencia Artificial mediante pip:

```bash
# Si tienes tarjeta gráfica NVIDIA (Soporte CUDA 12.1):
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
pip install faster-whisper transformers

# Si no tienes tarjeta gráfica (usando CPU):
pip install torch torchvision torchaudio
pip install faster-whisper transformers