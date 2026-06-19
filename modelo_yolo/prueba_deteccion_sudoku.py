from ultralytics import YOLO

# cargo mi modelo entrenado
modelo = YOLO(r'C:\Users\rober\Documents\Bootcamp\sudoku_master\app\modelos\modelo_yolo.pt')

# defino la ruta de la imagen
ruta_imagen = r'C:\Users\rober\Downloads\Captura de pantalla 2026-06-17 152439.png'

# hago la predicción sobre la imagen
results = modelo(ruta_imagen)

# muestro la imagen con la detección dibujada   
results[0].show()

# guardo el resultado
results[0].save()