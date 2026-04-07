# 🎬 Traductor de Videos IA — V5 (Local y Automático)

Traduce automáticamente el audio de cualquier video a subtítulos `.srt` en español usando IA 100% local. Si el video es en japonés, genera además un archivo `.srt` con la letra en **Romaji** (transliteración fonética al alfabeto latino).

---

## ¿Qué hace?

1. **Detecta el idioma** del video automáticamente (18 idiomas soportados)
2. **Transcribe el audio** completo usando Whisper (IA de OpenAI)
3. **Traduce al español** usando NLLB (IA de Meta/Facebook)
4. **Si es japonés**: genera un `.srt` extra con la letra en Romaji para seguir la canción

### Idiomas soportados

| Idioma     | Código | Idioma     | Código |
|------------|--------|------------|--------|
| Inglés     | en     | Español    | es     |
| Japonés    | ja     | Coreano    | ko     |
| Francés    | fr     | Chino      | zh     |
| Alemán     | de     | Ruso       | ru     |
| Italiano   | it     | Portugués  | pt     |
| Árabe      | ar     | Hindi      | hi     |
| Turco      | tr     | Polaco     | pl     |
| Holandés   | nl     | Sueco      | sv     |
| Indonesio  | id     | Vietnamita | vi     |

---

## Estructura de carpetas

```
tu-carpeta/
├── traductor_videos.py              ← Script principal
├── 1_Video_Original/                ← Pon aquí tu video (.mp4, .mkv, etc.)
└── 2_Subtitulos_Traducidos/         ← Aquí aparecen los .srt generados
    ├── MiVideo_ES.srt               ← Subtítulos en español
    └── MiVideo_ROMAJI.srt           ← Letra en Romaji (solo si es japonés)
```

Las carpetas se crean automáticamente en la primera ejecución.

---

## Instalación (Windows)

### Paso 1 — Python
Descargar de [python.org](https://www.python.org/downloads/) y marcar la casilla **"Add Python.exe to PATH"** en el instalador.

### Paso 2 — FFmpeg
Abrir CMD **como Administrador** y ejecutar:
```
winget install ffmpeg
```

### Paso 3 — Librerías de IA
Abrir CMD normal y ejecutar los siguientes comandos:

**Con tarjeta gráfica NVIDIA (recomendado):**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Sin tarjeta gráfica NVIDIA (usa CPU, más lento):**
```
pip install torch torchvision torchaudio
```

### Paso 4 — Modelos de transcripción y traducción
```
pip install faster-whisper transformers
```

### Paso 5 — Romaji para japonés (opcional)
```
pip install cutlet unidic-lite
```
> Esta librería permite generar subtítulos con la letra transliterada al alfabeto latino (Romaji) cuando el video es en japonés. Si no la instalas, el script funciona igual pero solo genera la traducción al español.

---

## Uso

1. Coloca tu video en la carpeta `1_Video_Original/`
2. Ejecuta el script:
```
python traductor_videos.py
```
3. Los subtítulos aparecerán en `2_Subtitulos_Traducidos/`

---

## ¿Qué es el Romaji?

El japonés utiliza tres sistemas de escritura: **Hiragana** (ひらがな), **Katakana** (カタカナ) y **Kanji** (漢字). El Romaji es la representación de estos caracteres en el alfabeto latino, lo que permite leer fonéticamente el japonés sin conocer los caracteres.

**Ejemplo:**
| Japonés             | Romaji                          | Español              |
|---------------------|---------------------------------|----------------------|
| 残酷な天使のテーゼ    | zankoku na tenshi no tēze       | La tesis del ángel cruel |
| 今日はとても暑い      | kyou wa totemo atsui            | Hoy hace mucho calor |

Esto es especialmente útil para seguir la letra de canciones japonesas (anime, J-Pop, J-Rock).

---

## Notas técnicas

- **Modelo Whisper**: Se usa `medium` por defecto. Si tienes una GPU potente (8GB+ VRAM), puedes cambiar a `large-v3` en el script para mayor precisión.
- **Filtro VAD**: Activado para filtrar silencios e instrumentación, mejora la precisión en videos musicales.
- **Tolerancia a errores**: Si un segmento falla, el script continúa con el siguiente sin perder el progreso.
- **Flush al disco**: Cada subtítulo se graba inmediatamente al archivo, así no pierdes progreso si el proceso se interrumpe.

---

## Comandos Git para actualizar

```bash
git add .
git commit -m "V5: Soporte Romaji japonés, filtro VAD, manejo de errores"
git push
```
