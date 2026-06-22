import time
import copy
import numpy as np
from pathlib import Path
from tensorflow import keras
from funciones import (
    verificar_rutas,
    cargar_modelos,
    aislar_y_corregir_sudoku,
    extraer_81_celdas,
    digitalizar_tablero,
    imprimir_tablero,
    backtrack,
    resolver_con_cnn
)

def ejecutar_flujo_completo():
    print("=== PIPELINE DE DETECCIÓN Y RESOLUCIÓN DE SUDOKUS ===")
    
    # Configuración de rutas relativas basadas en tu estructura
    RUTA_YOLO = Path('../modelos/yolo.pt') 
    RUTA_MODELO_OCR = Path('../modelos/modelo_ocr.keras')  
    RUTA_SOLVER = Path('../modelos/modelo_sudoku_mejor.keras')
    RUTA_IMAGEN = Path('../img_pruebas/024.png')       
    
    # 1. Validación de archivos inicial
    print("\n[Paso 1] Verificando ficheros locales...")
    if not verificar_rutas(RUTA_YOLO, RUTA_MODELO_OCR, RUTA_IMAGEN):
        print("❌ Error: Faltan archivos esenciales para el flujo. Abortando.")
        return

    # 2. Carga de los modelos de Visión Artificial
    print("\n[Paso 2] Cargando modelos de Redes Neuronales...")
    modelo_yolo, modelo_ocr = cargar_modelos(RUTA_YOLO, RUTA_MODELO_OCR)
    print('✓ Modelos de detección listos.')

    # 3. Segmentación del recuadro del Sudoku
    print(f"\n[Paso 3] Procesando imagen y ajustando perspectiva: {RUTA_IMAGEN.name}...")
    img_corregida = aislar_y_corregir_sudoku(RUTA_IMAGEN, modelo_yolo)
    if img_corregida is None:
        return

    # 4. Extracción de sub-imágenes limpias (sin líneas)
    print("\n[Paso 4] Extrayendo y limpiando las 81 celdas del tablero...")
    celdas = extraer_81_celdas(img_corregida)

    # 5. Digitalización mediante la CNN (OCR)
    print("\n[Paso 5] Identificando números con CNN OCR...")
    sudoku_detectado = digitalizar_tablero(celdas, modelo_ocr)
    
    n_pistas = sum(1 for f in sudoku_detectado for x in f if x != 0)
    print(f'✓ {n_pistas} pistas detectadas · {81 - n_pistas} huecos encontrados.')
    print("\nTablero Inicial Detectado:")
    imprimir_tablero(sudoku_detectado)

    # 6. Resolución Híbrida: Método A (CNN Solver)
    print("\n[Paso 6] Intentando resolución instantánea con CNN Solver...")
    if RUTA_SOLVER.exists():
        modelo_solver = keras.models.load_model(str(RUTA_SOLVER))
        
        t0_cnn = time.time()
        tablero_resuelto_cnn = resolver_con_cnn(sudoku_detectado, modelo_solver)
        t_cnn = (time.time() - t0_cnn) * 1000
        
        print(f"✓ CNN Solver ejecutada en {t_cnn:.1f} ms")
    else:
        print("⚠️ Modelo solver no encontrado, saltando predicción de red.")
        tablero_resuelto_cnn = None

    # 7. Resolución Híbrida: Método B (Backtracking con garantía total)
    print("\n[Paso 7] Ejecutando Backtracking...")
    tablero_bt = copy.deepcopy(sudoku_detectado)
    
    t0_bt = time.time()
    exito_bt = backtrack(tablero_bt)
    t_bt = (time.time() - t0_bt) * 1000

    if exito_bt:
        print(f"✓ Backtracking finalizado con éxito en {t_bt:.1f} ms")
        print("\nTablero Resuelto Final (Garantizado):")
        imprimir_tablero(tablero_bt)
    else:
        print("❌ El algoritmo de Backtracking determinó que el tablero no tiene solución.")
        print("   (Es muy probable que haya algún error de lectura en las predicciones del OCR)")

    # 8. Comparativa de rendimiento (Sección estrella para defender en la presentación)
    if tablero_resuelto_cnn and exito_bt:
        arr_bt = np.array(tablero_bt)
        arr_cnn = np.array(tablero_resuelto_cnn)
        arr_orig = np.array(sudoku_detectado)
        
        mask_huecos = (arr_orig == 0)
        n_huecos = mask_huecos.sum()
        
        aciertos_cnn = np.sum(arr_cnn[mask_huecos] == arr_bt[mask_huecos])
        pct_cnn = (aciertos_cnn / n_huecos) * 100
        
        print('\n' + '═' * 55)
        print(f'  {"MÉTODO":<25}{\"CELDAS CORRECTAS\":<20}{\"TIEMPO\"}')
        print('─' * 55)
        print(f'  {"CNN Solver (Red Neuronal)":<25}{f"{aciertos_cnn}/{n_huecos} ({pct_cnn:.1f}%)":<20}{t_cnn:>7.1f} ms')
        print(f'  {"Backtracking (Algoritmo)":<25}{f"{n_huecos}/{n_huecos} (100.0%)":<20}{t_bt:>7.1f} ms')
        print('═' * 55)
        
        if aciertos_cnn == n_huecos:
            print('\n✅ La CNN resolvió el puzzle perfectamente sin fallar un solo dígito.')
        else:
            print(f'\n⚠️ La CNN erró en {n_huecos - aciertos_cnn} celda(s). El Backtracking actuó de salvavidas corrigiendo los fallos.')

if __name__ == "__main__":
    ejecutar_flujo_completo()