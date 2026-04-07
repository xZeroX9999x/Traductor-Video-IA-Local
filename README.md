# Traductor de Videos IA — V5.2 (Local y Automatico)

Traduce automaticamente el audio de cualquier video a subtitulos `.srt` en espanol usando IA 100% local. Si el video es en japones, genera ademas un archivo `.srt` con la letra en **Romaji** (transliteracion fonetica al alfabeto latino).

---

## Que hace?

1. **Detecta el idioma** del video (18 idiomas soportados + seleccion manual)
2. **Transcribe el audio** completo usando Whisper (IA de OpenAI)
3. **Traduce al espanol** usando NLLB (IA de Meta/Facebook)
4. **Si es japones**: genera `.srt` con texto original + Romaji
5. **Code-switching**: detecta automaticamente cuando el vocalista cambia de japones a ingles dentro de la misma cancion

### Idiomas soportados

| Idioma     | Codigo | Idioma     | Codigo |
|------------|--------|------------|--------|
| Ingles     | en     | Espanol    | es     |
| Japones    | ja     | Coreano    | ko     |
| Frances    | fr     | Chino      | zh     |
| Aleman     | de     | Ruso       | ru     |
| Italiano   | it     | Portugues  | pt     |
| Arabe      | ar     | Hindi      | hi     |
| Turco      | tr     | Polaco     | pl     |
| Holandes   | nl     | Sueco      | sv     |
| Indonesio  | id     | Vietnamita | vi     |

---

## Estructura de carpetas

```
tu-carpeta/
├── traductor_videos.py              <- Script principal
├── 1_Video_Original/                <- Pon aqui tu video (.mp4, .mkv, etc.)
└── 2_Subtitulos_Traducidos/         <- Aqui aparecen los .srt generados
    ├── MiVideo_ES.srt               <- Subtitulos en espanol
    ├── MiVideo_JA.srt               <- Texto original japones (si aplica)
    └── MiVideo_ROMAJI.srt           <- Letra en Romaji (si aplica)
```

Las carpetas se crean automaticamente en la primera ejecucion.

---

## Instalacion (Windows)

### Paso 1 — Python
Descargar de [python.org](https://www.python.org/downloads/) y marcar la casilla **"Add Python.exe to PATH"** en el instalador.

### Paso 2 — FFmpeg
Abrir CMD **como Administrador** y ejecutar:
```
winget install ffmpeg
```

### Paso 3 — Librerias de IA
Abrir CMD normal y ejecutar los siguientes comandos:

**Con tarjeta grafica NVIDIA (recomendado):**
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Sin tarjeta grafica NVIDIA (usa CPU, mas lento):**
```
pip install torch torchvision torchaudio
```

### Paso 4 — Modelos de transcripcion y traduccion
```
pip install faster-whisper transformers
```

### Paso 5 — Romaji para japones (opcional)
```
pip install cutlet unidic-lite
```
> Esta libreria permite generar subtitulos con la letra transliterada al alfabeto latino (Romaji) cuando el video es en japones. Si no la instalas, el script funciona igual pero solo genera la traduccion al espanol.

---

## Uso

1. Coloca tu video en la carpeta `1_Video_Original/`
2. Ejecuta el script:
```
python traductor_videos.py
```
3. Elige el idioma del video (o deja que la IA lo detecte)
4. Indica si es un video musical o dialogo
5. Los subtitulos apareceran en `2_Subtitulos_Traducidos/`

---

## Code-switching (Japones + Ingles)

Muchas canciones japonesas (J-Rock, Visual Kei, J-Pop, anime) mezclan japones e ingles en la misma cancion. El script detecta esto automaticamente:

**Como funciona:**
- Al elegir "Japones" + "Si es musica", el script NO fuerza el idioma
- En vez de eso, usa un prompt inicial en japones para guiar a Whisper
- Whisper puede asi detectar cuando cambia a ingles
- Cada segmento se analiza por sus caracteres Unicode (kanji/kana vs latino)
- Los segmentos japoneses se traducen desde japones y se romanizan
- Los segmentos ingleses se traducen desde ingles (mejor calidad)

**Ejemplo de salida en consola:**
```
  [00:00:15,200 -> 00:00:18,900] [JA]
     Original: 闇の中で叫ぶ声が
     Romaji:   yami no naka de sakebu koe ga
     Espanol:  La voz que grita en la oscuridad

  [00:00:19,100 -> 00:00:22,400] [EN]
     Original: Destroy the world with fire
     Espanol:  Destruye el mundo con fuego
```

---

## Que es el Romaji?

El japones utiliza tres sistemas de escritura: **Hiragana**, **Katakana** y **Kanji**. El Romaji es la representacion de estos caracteres en el alfabeto latino, lo que permite leer foneticamente el japones sin conocer los caracteres.

**Ejemplo:**
| Japones               | Romaji                          | Espanol                    |
|-----------------------|---------------------------------|----------------------------|
| 残酷な天使のテーゼ      | zankoku na tenshi no teze       | La tesis del angel cruel   |
| 今日はとても暑い        | kyou wa totemo atsui            | Hoy hace mucho calor       |

Esto es especialmente util para seguir la letra de canciones japonesas (anime, J-Pop, J-Rock, Visual Kei).

---

## Modos de operacion

| Modo           | VAD    | Uso ideal                              |
|----------------|--------|----------------------------------------|
| Dialogo        | ON     | Entrevistas, documentales, tutoriales  |
| Musica         | OFF    | Videos musicales, conciertos, anime    |

El modo musica desactiva el filtro VAD (Voice Activity Detection) que normalmente descarta las voces cantadas como si fueran ruido.

---

## Notas tecnicas

- **Modelo Whisper**: Se usa `medium` por defecto. Si tienes una GPU potente (8GB+ VRAM), puedes cambiar a `large-v3` en el script para mayor precision.
- **Tolerancia a errores**: Si un segmento falla, el script continua con el siguiente sin perder el progreso.
- **Flush al disco**: Cada subtitulo se graba inmediatamente al archivo.

---

## Comandos Git para actualizar

```bash
git add .
git commit -m "V5.2: Code-switching JA/EN, deteccion por segmento"
git push
```
