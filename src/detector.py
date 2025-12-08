from ultralytics import YOLO
import cv2
import time
import numpy as np

class SignDetector:
    def __init__(self, model_path, device="cpu", batch_size=16):
        """
        Detector de sinais usando YOLO
        Args:
            model_path: caminho para o modelo (.pt ou .onnx)
            device: "cpu" ou "cuda"
            batch_size: tamanho do batch para inferência
        """
        self.model = YOLO(model_path)
        self.device = device
        self.batch_size = batch_size
        self.inference_times = []

    def detect_batch(self, batch):
        """
        Processa um batch de frames de forma segura
        Args:
            batch: numpy array de shape (B, H, W, 3) - frames em BGR
        Returns:
            lista de resultados do YOLO para cada frame
        """
        results = []

        # Garantir que todos os frames estejam no formato correto
        batch_rgb = []
        for frame in batch:
            # Converte BGR -> RGB e garante que não há strides negativas
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).copy()
            batch_rgb.append(frame_rgb)

        start_time = time.time()
        try:
            # YOLO pode receber uma lista de imagens para inferência em batch
            batch_results = self.model(batch_rgb, device=self.device, verbose=False)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)

            # batch_results é uma lista com resultados para cada frame
            results.extend(batch_results)
        except Exception as e:
            print(f"ERRO na inferência: {e}")
        
        return results

    def get_performance_stats(self):
        """Retorna estatísticas de performance"""
        if not self.inference_times:
            return {
                "avg_inference_time": 0,
                "total_batches": 0,
                "fps_per_batch": 0,
                "total_frames": 0
            }

        avg_time = sum(self.inference_times) / len(self.inference_times)
        fps = self.batch_size / avg_time if avg_time > 0 else 0

        return {
            "avg_inference_time": avg_time,
            "total_batches": len(self.inference_times),
            "fps_per_batch": fps,
            "total_frames": len(self.inference_times) * self.batch_size
        }
