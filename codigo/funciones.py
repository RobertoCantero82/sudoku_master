import copy
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from tensorflow import keras

# CARGA DE MODELOS Y COMPROBACIONES

def verificar_rutas(ruta_yolo, ruta_ocr, ruta_imagen):
    """Comprueba la existencia de los archivos críticos antes de iniciar."""
    yolo_ok = Path(ruta_yolo).exists()
    ocr_ok = Path(ruta_ocr).exists()
    img_ok = Path(ruta_imagen).exists()
    
    print(f'  YOLO:      {ruta_yolo}  ({"existe" if yolo_ok else "NO ENCONTRADO"})')
    print(f'  CNN OCR:   {ruta_ocr}  ({"existe" if ocr_ok else "NO ENCONTRADO"})')
    print(f'  Imagen:    {ruta_imagen}  ({"existe" if img_ok else "NO ENCONTRADA"})')
    
    return yolo_ok and ocr_ok and img_ok

def cargar_modelos(ruta_yolo, ruta_ocr):
    """Cargo el modelo YOLO y la CNN de clasificación."""
    modelo_yolo = YOLO(str(ruta_yolo))
    modelo_ocr = keras.models.load_model(str(ruta_ocr))
    return modelo_yolo, modelo_ocr

# PROCESAMIENTO DE IMAGEN Y PERSPECTIVA

def ordenar_esquinas(pts):
    """Ordena las esquinas en el orden específico: [TL, TR, BR, BL]."""
    pts = pts.reshape(4, 2).astype('float32')
    rect = np.zeros((4, 2), dtype='float32')

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]       # top-left
    rect[2] = pts[np.argmax(s)]       # bottom-right

    d = np.diff(pts, axis=1).flatten()
    rect[1] = pts[np.argmin(d)]       # top-right
    rect[3] = pts[np.argmax(d)]       # bottom-left
    return rect

def aislar_y_corregir_sudoku(ruta_imagen, modelo_yolo):
    """Detecta el tablero con YOLO y corrige su perspectiva a 450x450."""
    imagen_original = cv2.imread(str(ruta_imagen))
    if imagen_original is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen en {ruta_imagen}")
        
    results = modelo_yolo(imagen_original, verbose=False)
    
    if len(results[0].boxes) == 0:
        print('❌ No se detectó ningún recuadro de sudoku')
        return None

    boxes = results[0].boxes.xyxy.cpu().numpy()
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    idx = int(np.argmax(areas))
    
    x1, y1, x2, y2 = map(int, boxes[idx])
    img_recortada = imagen_original[y1:y2, x1:x2]
    
    # Flujo de corrección de perspectiva interno
    gris = cv2.cvtColor(img_recortada, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gris, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 4)

    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    contornos, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        print('⚠️ No se encontraron contornos, usando imagen sin corregir')
        return img_recortada

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)
    img_area = img_recortada.shape[0] * img_recortada.shape[1]

    for c in contornos[:8]:
        if cv2.contourArea(c) < img_area * 0.15:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        if len(approx) == 4:
            rect = ordenar_esquinas(approx.reshape(4, 2))
        else:
            box = cv2.boxPoints(cv2.minAreaRect(c))
            rect = ordenar_esquinas(box)

        lado = 450
        dst = np.array([[0, 0], [lado, 0], [lado, lado], [0, lado]], dtype='float32')
        M = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(img_recortada, M, (lado, lado))

    print('⚠️ No se encontró cuadrícula clara, usando imagen sin corregir')
    return img_recortada

# SEGMENTACIÓN Y ELIMINACIÓN DE LÍNEAS

def eliminar_lineas(warped):
    """Borra las líneas de la cuadrícula mediante morfología matemática."""
    gris = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    binaria = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 5)

    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (warped.shape[1] // 9, 1))
    h_lines = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, h_kernel)

    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, warped.shape[0] // 9))
    v_lines = cv2.morphologyEx(binaria, cv2.MORPH_OPEN, v_kernel)

    lineas = cv2.dilate(cv2.add(h_lines, v_lines), np.ones((3, 3), np.uint8), iterations=1)
    resultado = gris.copy()
    resultado[lineas > 0] = 255
    return resultado

def extraer_81_celdas(img_corregida, margen=6):
    """Divide el tablero limpio en una lista indexada de 9x9 celdas."""
    img_sin_lineas = eliminar_lineas(img_corregida)
    alto, ancho = img_sin_lineas.shape[:2]
    celda_alto = alto // 9
    celda_ancho = ancho // 9

    celdas = []
    for fila in range(9):
        fila_celdas = []
        for col in range(9):
            y1c = fila * celda_alto + margen
            y2c = y1c + celda_alto - margen * 2
            x1c = col * celda_ancho + margen
            x2c = x1c + celda_ancho - margen * 2
            fila_celdas.append(img_sin_lineas[y1c:y2c, x1c:x2c])
        celdas.append(fila_celdas)
    return celdas

# RECONOCIMIENTO OCR DE DÍGITOS

def celda_tiene_digito(celda):
    """Heurística basada en contraste para comprobar si la celda está vacía."""
    gris = celda if len(celda.shape) == 2 else cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris.astype(np.float32), (48, 48))
    e = 8

    esquinas = np.concatenate([gris[:e, :e].ravel(), gris[:e, -e:].ravel(),
                               gris[-e:, :e].ravel(), gris[-e:, -e:].ravel()])
    fondo = np.median(esquinas)
    centro = gris[12:36, 12:36]

    return (np.sum(centro < fondo - 40) / centro.size) > 0.03

def preprocesar_para_cnn(celda):
    """Prepara la celda en formato binario 28x28 normalizado [0, 1]."""
    gris = celda if len(celda.shape) == 2 else cv2.cvtColor(celda, cv2.COLOR_BGR2GRAY)
    gris = cv2.resize(gris, (28, 28), interpolation=cv2.INTER_AREA)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    gris = clahe.apply(gris)
    _, binaria = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaria.astype('float32') / 255.0

def predecir_celda(celda, modelo_ocr):
    """Clasifica el dígito de la celda (0 si está vacía, 1-9 si contiene número)."""
    if not celda_tiene_digito(celda):
        return 0

    img = preprocesar_para_cnn(celda).reshape(1, 28, 28, 1)
    pred = modelo_ocr.predict(img, verbose=0)[0]
    clase = int(np.argmax(pred))

    if clase == 0:
        clase = int(np.argmax(pred[1:])) + 1
    return clase

def digitalizar_tablero(celdas, modelo_ocr):
    """Construye la matriz numérica 9x9 procesando las 81 celdas."""
    sudoku = []
    for fila in range(9):
        fila_nums = []
        for col in range(9):
            fila_nums.append(predecir_celda(celdas[fila][col], modelo_ocr))
        sudoku.append(fila_nums)
    return sudoku

# RESOLUCIÓN (BACKTRACKING & CNN)


def es_valido(t, fila, col, num):
    """Valida las reglas tradicionales del Sudoku."""
    if num in t[fila]: return False
    if num in [t[i][col] for i in range(9)]: return False

    bf, bc = (fila // 3) * 3, (col // 3) * 3
    for i in range(bf, bf + 3):
        for j in range(bc, bc + 3):
            if t[i][j] == num: return False
    return True

def backtrack(t):
    """Resuelve el tablero in-place usando fuerza bruta inteligente."""
    for i in range(9):
        for j in range(9):
            if t[i][j] == 0:
                for num in range(1, 10):
                    if es_valido(t, i, j, num):
                        t[i][j] = num
                        if backtrack(t): return True
                        t[i][j] = 0
                return False
    return True

def resolver_con_cnn(sudoku_matriz, modelo_solver):
    """Usa el modelo de Deep Learning para predecir el tablero completo de una vez."""
    puzzle_flat = np.array([v for fila in sudoku_matriz for v in fila], dtype='float32')
    puzzle_norm = puzzle_flat / 9.0
    puzzle_input = puzzle_norm.reshape(1, 81)
    
    pred_raw = modelo_solver.predict(puzzle_input, verbose=0)[0]
    pred_clases = np.argmax(pred_raw, axis=1)
    digitos_cnn = pred_clases + 1
    
    sol_cnn = np.where(puzzle_flat != 0, puzzle_flat.astype(int), digitos_cnn)
    return sol_cnn.reshape(9, 9).tolist()

# VISUALIZACIÓN POR CONSOLA


def imprimir_tablero(matriz, texto_vacio="·"):
    """Formatea e imprime de manera elegante el tablero en la terminal."""
    print('─' * 25)
    for i, fila in enumerate(matriz):
        if i % 3 == 0 and i != 0: 
            print('─' * 25)
        s = ''
        for j, num in enumerate(fila):
            if j % 3 == 0 and j != 0: 
                s += '│ '
            s += f'{num if num != 0 else texto_vacio} '
        print(s)