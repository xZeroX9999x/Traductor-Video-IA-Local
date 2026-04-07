# Traductor de Videos IA — V6 (Separacion Vocal + Romaji)

Traduce automaticamente el audio de cualquier video a subtitulos `.srt` en espanol usando IA 100% local.

Para **videos musicales**, separa la voz de los instrumentos antes de transcribir, lo que mejora drasticamente la precision. Si el video es en japones, genera ademas Romaji (transliteracion fonetica).

---

## Pipeline completo

```
Video musical (.mp4)
  |
  v
[FFmpeg] Extraer audio -> audio.wav
  |
  v
[Demucs IA] Separar voz -> vocals.wav (sin guitarras, bateria, bajo)
  |
  v
[Whisper IA] Transcribir voz limpia -> texto original
  |
  v
[Deteccion] Japones o ingles? (por segmento, Unicode)
  |
  v
[NLLB IA] Traducir al espanol (desde idioma correcto)
  |
  v
[cutlet] Generar Romaji (si es japones)
  |
  v
Archivos .srt finales
```

---

## Archivos generados

| Archivo               | Contenido                              | Cuando se genera         |
|-----------------------|----------------------------------------|--------------------------|
| `Video_ES.srt`        | Subtitulos en espanol                  | Siempre                  |
| `Video_JA.srt`        | Texto original en kanji/kana           | Solo si es japones       |
| `Video_ROMAJI.srt`    | Letra en alfabeto latino (fonetica)    | Solo si es japones       |

---

## Instalacion (Windows)

### Paso 1 — Python
Descargar de [python.org](https://www.python.org/downloads/) y marcar **"Add Python.exe to PATH"**.

### Paso 2 — FFmpeg
CMD como Administrador:
```
winget install ffmpeg
```

### Paso 3 — PyTorch
CMD normal:

Con GPU NVIDIA:
```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Sin GPU (CPU, mas lento):
```
pip install torch torchvision torchaudio
```

### Paso 4 — Modelos de transcripcion y traduccion
```
pip install faster-whisper transformers
```

### Paso 5 — Separacion vocal para musica (recomendado)
```
pip install demucs
```
> Demucs (de Meta/Facebook) separa la voz de los instrumentos. Sin esto, Whisper escucha guitarras, bateria y voz todo mezclado y la transcripcion sale incorrecta. Con Demucs, Whisper recibe solo la voz limpia.

### Paso 6 — Romaji para japones (opcional)
```
pip install cutlet unidic-lite
```
> Genera subtitulos con la letra transliterada al alfabeto latino.

---

## Uso

1. Coloca tu video en `1_Video_Original/`
2. Ejecuta:
```
python traductor_videos.py
```
3. Elige el idioma y si es video musical
4. Resultados en `2_Subtitulos_Traducidos/`

---

## Code-switching (Japones + Ingles)

Muchas canciones japonesas mezclan japones e ingles. El script detecta esto automaticamente analizando los caracteres Unicode de cada segmento:

```
  [00:00:15,200 -> 00:00:18,900] [JA]
     Original: 闇の中で叫ぶ声が
     Romaji:   yami no naka de sakebu koe ga
     Espanol:  La voz que grita en la oscuridad

  [00:00:19,100 -> 00:00:22,400] [EN]
     Original: Destroy the world with fire
     Espanol:  Destruye el mundo con fuego
```

Cada segmento se traduce desde su idioma real (JA o EN), mejorando la calidad.

---

## Que es el Romaji?

El japones usa tres escrituras: **Hiragana**, **Katakana** y **Kanji**. El Romaji los representa en alfabeto latino:

| Japones               | Romaji                          | Espanol                    |
|-----------------------|---------------------------------|----------------------------|
| 残酷な天使のテーゼ      | zankoku na tenshi no teze       | La tesis del angel cruel   |
| 闇の中で叫ぶ            | yami no naka de sakebu          | Gritar en la oscuridad     |

---

## Estructura de carpetas

```
tu-carpeta/
├── traductor_videos.py
├── 1_Video_Original/          <- Pon tu video aqui
├── 2_Subtitulos_Traducidos/   <- SRTs generados
└── 3_Temp/                    <- Se crea y borra automaticamente
```

---

## Notas tecnicas

- **Modelo Whisper**: `medium` por defecto. Cambiar a `large-v3` en el script si tienes GPU con 8GB+ VRAM.
- **Demucs**: La primera ejecucion descarga el modelo (~80MB). Las siguientes son mas rapidas.
- **Tolerancia a errores**: Si un segmento falla, continua con el siguiente.
- **Sin Demucs**: El script funciona igual, pero la precision en musica sera menor.

---

## Git

```bash
git add .
git commit -m "V6: Separacion vocal Demucs, code-switching JA/EN, Romaji"
git push
```
