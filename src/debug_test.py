from ultralytics import YOLO
import cv2
from visualizer import ResultVisualizer   # sua classe

# carregando modelo treinado
model = YOLO("models/best.pt")

# carregando a imagem
img = cv2.imread("data/no-parking.jpg")

# inferência
results = model(img)

# visualizador
viz = ResultVisualizer()

# desenhar caixas — usar results[0]
img_out = viz.draw_boxes(img, results[0])

# salvar no lugar de exibir
cv2.imwrite("resultado_parki.jpg", img_out)
print("Imagem salva como resultado_stop.jpg")
