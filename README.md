# 🟢 SUDOKU RELOADED

> *Hay un fallo en la Matrix.* Sube una foto de un sudoku y el sistema lo localiza, descifra y resuelve: visión por computador, deep learning y backtracking, envueltos en una interfaz que te mete dentro de la Matrix.

Proyecto final del bootcamp de Data Science de **The Bridge** (2026).

---

## 🎯 Qué hace

Subes una foto de un sudoku (captura de pantalla, foto de periódico o de libro) y la aplicación:

1. **Localiza** el recuadro del sudoku en la imagen con un modelo YOLO entrenado a medida.
2. **Recorta y corrige** la perspectiva para obtener una cuadrícula limpia.
3. **Divide** la cuadrícula en 81 celdas.
4. **Lee** el dígito de cada celda con una red neuronal (OCR).
5. **Resuelve** el puzzle por backtracking.
6. Muestra la solución diferenciando los números originales de los reconstruidos por el sistema.

---

## 🧠 Arquitectura del pipeline

```
Foto → YOLO (detección) → Recorte + perspectiva → 81 celdas → OCR (CNN) → Matriz 9×9 → Backtracking → Solución
```

| Fase | Técnica | Detalle |
|------|---------|---------|
| Detección del recuadro | **YOLO** (Ultralytics) | Modelo entrenado con ~230 imágenes etiquetadas. Confianza ~98%. |
| Corrección de perspectiva | **OpenCV** | `adaptiveThreshold` + contorno de la cuadrícula interior + `getPerspectiveTransform`. Ignora bordes de color (p. ej. el borde azul de cuadernos). |
| División en celdas | **OpenCV** | Rejilla 9×9 con margen proporcional para evitar las líneas. |
| Lectura de dígitos (OCR) | **CNN (Keras)** | Red entrenada con **dígitos impresos generados a partir de fuentes tipográficas** (no MNIST manuscrito) + *data augmentation*. ~100% de accuracy. |
| Resolución | **Backtracking** | Algoritmo clásico de fuerza bruta con poda. Garantiza solución correcta. |
| Resolución alternativa | **CNN Solver** | Red convolucional que predice la solución completa "por intuición" (en desarrollo, para comparar velocidad vs fiabilidad). |

---

## 🗂️ Estructura del proyecto

```
sudoku_master/
├── app/                  # Recursos de la app
├── cuadernos/            # Notebooks de desarrollo y entrenamiento
│   ├── entrenar_ocr_fuentes.ipynb     # Entrena la CNN OCR (dígitos de fuentes)
│   ├── entrenar_solver_colab.ipynb    # Entrena la CNN Solver (Colab + GPU)
│   └── sudoku_local_v2.ipynb          # Pipeline completo paso a paso (debug)
├── img_pruebas/          # Imágenes de prueba
├── modelos/
│   ├── yolo.pt                  # Detección del recuadro
│   ├── modelo_ocr.keras         # OCR de dígitos
│   └── modelo_sudoku.keras      # CNN Solver
├── app.py                # Aplicación Streamlit (tema Matrix)
├── packages.txt          # Dependencias del sistema (libGL, etc.)
├── requirements.txt      # Dependencias Python
└── README.md
```

---

## 🚀 Puesta en marcha (local)

### 1. Clonar y crear entorno virtual

```bash
git clone https://github.com/RobertoCantero82/sudoku_master.git
cd sudoku_master

python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# macOS / Linux
source .venv/bin/activate
```

> Si en Windows PowerShell te bloquea la activación:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Lanzar la app

```bash
streamlit run app.py
```

La primera vez, EasyOCR/Keras descargan sus pesos (puede tardar un par de minutos).

---

## 🖥️ La app: SUDOKU RELOADED

Interfaz con estética *Matrix*: lluvia de código verde de fondo, terminal en verde fósforo y un flujo por fases que el usuario va ejecutando con botones:

- 🔴 **TOMAR LA PÍLDORA ROJA** — inicia la intrusión
- ▶ **FASE 1 · LOCALIZACIÓN SUDOKU** — YOLO detecta y recorta
- ▶ **FASE 2 · DESENCRIPTANDO** — OCR lee los dígitos
- ▶ **FASE 3 · HACKEANDO MATRIX** — backtracking resuelve

Durante cada fase aparecen mensajes animados de "hackeo" de broma, y la solución final distingue en **verde** los dígitos originales y en **ámbar** los reconstruidos por el sistema.

---

## 🏋️ Entrenamiento de los modelos

### CNN OCR (dígitos)

`cuadernos/entrenar_ocr_fuentes.ipynb` genera un dataset de dígitos a partir de las fuentes tipográficas del sistema (Arial, Times, Courier, DejaVu, etc.) con *data augmentation* (rotaciones, desenfoque, ruido, sombras) para simular fotos reales. La clave del proyecto: **MNIST manuscrito no sirve para dígitos impresos**; generar el dataset desde fuentes da casi 100% de acierto.

### CNN Solver

`cuadernos/entrenar_solver_colab.ipynb` (pensado para Colab + GPU) entrena una red convolucional con un dataset de >1 millón de sudokus. Métrica clave: **% de sudokus resueltos completos** (no solo accuracy por celda). La narrativa del proyecto compara la red neuronal (rápida pero falible) contra el backtracking (lento pero 100% fiable).

---

## 🛠️ Stack tecnológico

- **Python**
- **Ultralytics YOLO** — detección de objetos
- **OpenCV** — procesamiento de imagen
- **TensorFlow / Keras** — redes neuronales (OCR y solver)
- **EasyOCR** — OCR de respaldo
- **Streamlit** — interfaz web
- **NumPy**, **Pillow**, **Matplotlib**

---

## 📋 Notas

- Para mejores resultados, sube la foto **bien orientada** (números legibles) y con buena iluminación. Las capturas de pantalla dan lectura prácticamente perfecta; las fotos con sombras fuertes pueden generar algún error de lectura.
- Si el backtracking falla, suele indicar un error de lectura del OCR: prueba con otra imagen.

---

## 👤 Autor

**Roberto Cantero** — Periodista de tecnología y ciencia reconvertido a Data Science.
GitHub: [@RobertoCantero82](https://github.com/RobertoCantero82)

---

*"REALITY.EXE HA DEJADO DE FUNCIONAR · RESUELVE EL SUDOKU"* 🟢
