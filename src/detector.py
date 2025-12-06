from ultralytics import YOLO
import time

class SignDetector:
    def __init__(self, model_path, device="cpu", batch_size=16):
        """
        Inicializa o detector com o modelo treinado
        Args:
            model_path: caminho para .pt ou .onnx
            device: "cpu" ou "cuda" ou 0 para GPU
            batch_size: tamanho do batch (importante para cálculo de FPS)
        """
        self.model = YOLO(model_path)
        self.device = device
        self.batch_size = batch_size  # ADICIONE ESTA LINHA
        self.inference_times = []
    
    def detect_batch(self, batch):
        """
        Processa um batch de frames
        Args:
            batch: numpy array de shape (batch_size, H, W, C)
        Returns:
            Lista de resultados para cada frame
        """
        if batch.size == 0:
            print("AVISO: Batch vazio recebido")
            return []
        
        start_time = time.time()
        
        # Converte batch para lista de imagens
        # O YOLO espera imagens em formato (H, W, C) - RGB
        batch_list = []
        for i in range(batch.shape[0]):
            frame = batch[i]
            # Converte BGR para RGB se necessário
            if frame.shape[2] == 3:
                frame_rgb = frame[..., ::-1]  # BGR to RGB
                batch_list.append(frame_rgb)
            else:
                batch_list.append(frame)
        
        # Executa inferência em BATCH
        try:
            results = self.model(batch_list, device=self.device, verbose=False)
            inference_time = time.time() - start_time
            self.inference_times.append(inference_time)
            
            print(f"  Batch processado em {inference_time:.3f}s")
            return results
        except Exception as e:
            print(f"ERRO na inferência: {e}")
            return []
    
    def get_performance_stats(self):
        """Retorna estatísticas de performance"""
        if not self.inference_times:
            return {
                "avg_inference_time": 0,
                "total_batches": 0,
                "fps_per_batch": 0,  # ADICIONE ESTA CHAVE
                "total_frames": 0
            }
        
        avg_time = sum(self.inference_times) / len(self.inference_times)
        fps_per_batch = self.batch_size / avg_time if avg_time > 0 else 0
        
        return {
            "avg_inference_time": avg_time,
            "total_batches": len(self.inference_times),
            "fps_per_batch": fps_per_batch,  # AGORA ESTÁ DEFINIDA
            "total_frames": len(self.inference_times) * self.batch_size
        }