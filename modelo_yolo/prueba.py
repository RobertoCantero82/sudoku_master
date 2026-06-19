from ultralytics import YOLO

modelo = YOLO('yolo11n.pt')

modelo.train(
    # Archivo yaml para definir el dataset.
    data="data.yaml",
    epochs=10,  # Numero de epocas que defino.
    batch=64,   #Tamaño de batch.
    imgsz=640,  # Tamaño de las imagenes.
    project="sudoku",  # Carpeta donde se guardará el entrenamiento.
    name="sudoku",  # Nombre del modelo.
    save=True,      # Guardar el modelo después de entrenar.
    exist_ok=True   # Sobrescribir resultados si ya existe una carpeta.
)