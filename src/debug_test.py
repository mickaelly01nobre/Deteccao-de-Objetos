from ultralytics import YOLO
import cv2
from visualizer import ResultVisualizer   # sua classe

# carregando modelo treinado
model = YOLO("models/best.pt")

# carregando vídeo
cap = cv2.VideoCapture("data/Video_transito.mp4")

# verificar se abriu
if not cap.isOpened():
    print("Erro ao abrir o vídeo!")
    exit()

# visualizador
viz = ResultVisualizer()

# preparar o writer para salvar vídeo de saída
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter("resultado_video.mp4", fourcc, fps, (w, h))

# loop de leitura frame a frame
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # inferência por frame
    results = model(frame)

    # desenhar caixas
    frame_out = viz.draw_boxes(frame, results[0])

    # salvar no vídeo final
    out.write(frame_out)

# liberar recursos
cap.release()
out.release()

print("Vídeo salvo como resultado_video.mp4")
