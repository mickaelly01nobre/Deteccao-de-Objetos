import cv2
import numpy as np
from detector import SignDetector   # sua classe
import time

# ==========================
# CONFIGURAÇÕES
# ==========================
VIDEO_PATH = "data/Video_transito.mp4"       # coloque o caminho do seu vídeo
MODEL_PATH = "models/best.pt"       # seu modelo YOLO
DEVICE = "cpu"                      # ou "cuda" se tiver GPU
BATCH_SIZE = 16                     # pode ajustar

# ==========================
# INICIALIZA DETECTOR
# ==========================
detector = SignDetector(
    model_path=MODEL_PATH,
    device=DEVICE,
    batch_size=BATCH_SIZE
)

# ==========================
# LEITURA DO VÍDEO
# ==========================
cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    print("Erro ao abrir o vídeo.")
    exit()

batch_frames = []
total_frames_read = 0

print("\n--- PROCESSANDO VÍDEO ---\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames_read += 1
    batch_frames.append(frame)

    # Quando encher o batch, processa
    if len(batch_frames) == BATCH_SIZE:
        batch_np = np.array(batch_frames)
        detector.detect_batch(batch_np)
        batch_frames = []  # limpa para o próximo

# Processa últimos frames se o batch não estiver completo
if len(batch_frames) > 0:
    batch_np = np.array(batch_frames)
    detector.detect_batch(batch_np)

cap.release()

# ==========================
# ESTATÍSTICAS FINAIS
# ==========================
stats = detector.get_performance_stats()

print("\n======= RESULTADOS =======")
print(f"Total de frames lidos: {total_frames_read}")
print(f"Total de batches processados: {stats['total_batches']}")
print(f"Tempo médio por batch: {stats['avg_inference_time']:.3f}s")
print(f"FPS por batch: {stats['fps_per_batch']:.2f}")
print(f"Total de frames processados: {stats['total_frames']}")
print("==========================\n")
