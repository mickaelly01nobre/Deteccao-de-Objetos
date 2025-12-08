import cv2
from ultralytics import YOLO
from visualizer import ResultVisualizer

# Carrega modelo treinado
model = YOLO("models/yolo11n.pt")

# Carrega vídeo
video_path = "data/Video_transito.mp4"
cap = cv2.VideoCapture(video_path)

# Visualizador
viz = ResultVisualizer()

frames_processados = []  # onde vamos guardar os frames pós-detecção

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inferência frame a frame
    results = model(frame)

    # Desenhar boxes (sempre usar results[0])
    frame_out = viz.draw_boxes(frame, results[0])

    # Armazena o frame para gerar o vídeo final
    frames_processados.append(frame_out)

cap.release()

# Salvar vídeo final
viz.save_video(frames_processados, "resultado_video.mp4", fps=30)

print("Vídeo salvo como resultado_video.mp4")
