"""
 A classe SignDetector é um detector de sinais de trânsito usando YOLO com suporte a batch processing.
 Ela permite processar vários frames de uma vez (batch), otimizando a inferência e medindo
 o desempenho em tempo real. A classe fornece métodos para detectar em batches simples ou
 otimizados, processar um vídeo inteiro em batches, detectar frame único para debug/fallback,
 e registrar estatísticas detalhadas como FPS médio, tempo total de inferência, tempo por batch
 e por frame, além do total de frames processados. Ela também possui logs opcionais para monitorar
 a performance e faz pré-processamento automático, como redimensionamento de frames, para
 garantir compatibilidade com o modelo YOLO.
"""

from ultralytics import YOLO
import cv2
import time
import numpy as np
from typing import List, Tuple, Optional
import torch

class SignDetector:
    def __init__(self, model_path: str, device: str = "cpu", batch_size: int = 8, verbose: bool = False):
        """
        Detector de sinais usando YOLO otimizado
        
        Args:
            model_path: caminho para o modelo
            device: "cpu" ou "cuda"
            batch_size: tamanho do batch (recomendado: 4-8 para CPU, 16-32 para GPU)
            verbose: mostra logs do YOLO
        """
        self.model = YOLO(model_path)
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        self.inference_times = []
        self.frame_count = 0
        
        # Configurações para melhor performance
        self.model.overrides['verbose'] = verbose
        self.model.overrides['device'] = device
        self.model.overrides['conf'] = 0.25  # Limiar de confiança
        self.model.overrides['iou'] = 0.45   # Limiar de IOU para NMS
        
        print(f" Modelo carregado: {model_path}")
        print(f" Device: {device}")
        print(f" Batch size: {batch_size}")
    
    def detect_batch_simple(self, batch: List[np.ndarray]) -> List:
        """
        Método SIMPLES e FUNCIONAL para batch processing
        Usa a API nativa do Ultralytics que aceita lista de frames
        """
        if not batch:
            return []
        
        start_time = time.time()
        
        try:
            # Método mais direto: Ultralytics aceita lista de frames
            results = self.model(batch, verbose=self.verbose)
            
            # Converte Results objects para lista
            processed_results = list(results)
            
        except Exception as e:
            print(f"  ERRO no batch processing: {e}")
            print(" Usando fallback individual...")
            
            # Fallback: processamento frame por frame
            processed_results = []
            for frame in batch:
                try:
                    result = self.model(frame, verbose=False)[0]
                    processed_results.append(result)
                except Exception as frame_error:
                    print(f"     Erro no frame individual: {frame_error}")
                    processed_results.append(None)
        
        # Estatísticas
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += len(batch)
        
        return processed_results
    
    def detect_batch_optimized(self, batch: List[np.ndarray]) -> List:
        """
        Método OTIMIZADO para batch processing
        """
        if not batch:
            return []
        
        start_time = time.time()
        batch_size = len(batch)
        
        try:
            # Pré-processamento otimizado
            processed_batch = []
            for frame in batch:
                # Redimensiona se necessário
                if frame.shape[:2] != (640, 640):
                    frame = cv2.resize(frame, (640, 640))
                processed_batch.append(frame)
            
            # Usa a API do Ultralytics com lista de frames
            # O Ultralytics v8+ faz o batch processing automaticamente
            results = self.model(processed_batch, 
                               verbose=self.verbose,
                               imgsz=640,
                               device=self.device,
                               conf=0.25,
                               iou=0.45)
            
            processed_results = list(results)
            
        except Exception as e:
            print(f" Erro no batch otimizado: {e}")
            return self.detect_batch_simple(batch)
        
        # Estatísticas
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += batch_size
        
        # Log de performance
        fps = batch_size / inference_time if inference_time > 0 else 0
        if self.verbose and len(self.inference_times) % 5 == 0:
            print(f"📊 Batch {len(self.inference_times)}: {inference_time:.3f}s, FPS: {fps:.1f}")
        
        return processed_results
    
    def detect_batch(self, batch: List[np.ndarray]) -> List:
        """Alias para o método otimizado"""
        return self.detect_batch_optimized(batch)
    
    def detect_single(self, frame: np.ndarray):
        """Detecção em frame único (para debugging)"""
        start_time = time.time()
        
        result = self.model(frame, verbose=False)[0]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        return result
    
    def process_video_batches(self, video_path: str, max_frames: Optional[int] = None):
        """
        Processa vídeo completo em batches
        
        Returns:
            Lista de resultados para cada frame
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f" Não foi possível abrir vídeo: {video_path}")
            return []
        
        # Informações do vídeo
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f" Processando vídeo: {video_path}")
        print(f" Frames: {total_frames}, FPS: {fps:.2f}, Resolução: {width}x{height}")
        
        if max_frames and max_frames < total_frames:
            total_frames = max_frames
            print(f" Limitado aos primeiros {max_frames} frames")
        
        all_results = []
        current_batch = []
        frame_indices = []
        
        from tqdm import tqdm
        
        pbar = tqdm(total=total_frames, desc="Processando vídeo")
        
        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            
            current_batch.append(frame)
            frame_indices.append(frame_idx)
            
            # Processa quando batch está completo
            if len(current_batch) == self.batch_size:
                results = self.detect_batch(current_batch)
                all_results.extend(zip(frame_indices, results))
                
                current_batch = []
                frame_indices = []
            
            pbar.update(1)
        
        # Processa frames restantes
        if current_batch:
            results = self.detect_batch(current_batch)
            all_results.extend(zip(frame_indices, results))
        
        cap.release()
        pbar.close()
        
        # Ordena resultados por índice do frame
        all_results.sort(key=lambda x: x[0])
        return [result for _, result in all_results]
    
    def get_performance_stats(self) -> dict:
        """Retorna estatísticas detalhadas de performance"""
        if not self.inference_times:
            return {
                "total_frames": 0,
                "total_time": 0,
                "avg_fps": 0,
                "avg_inference_time": 0,
                "total_batches": 0,
                "frames_per_batch": 0
            }
        
        total_time = sum(self.inference_times)
        avg_inference = total_time / len(self.inference_times)
        avg_fps = self.frame_count / total_time if total_time > 0 else 0
        
        return {
            "total_frames": self.frame_count,
            "total_time": total_time,
            "avg_fps": avg_fps,
            "avg_inference_time": avg_inference,
            "total_batches": len(self.inference_times),
            "frames_per_batch": self.batch_size,
            "time_per_frame": total_time / self.frame_count if self.frame_count > 0 else 0
        }
    
    def print_statistics(self):
        """Imprime estatísticas formatadas"""
        stats = self.get_performance_stats()
        
        print("\n" + "="*50)
        print(" ESTATÍSTICAS DE PERFORMANCE")
        print("="*50)
        print(f"Total de frames processados: {stats['total_frames']}")
        print(f"Tempo total de inferência: {stats['total_time']:.2f}s")
        print(f"FPS médio: {stats['avg_fps']:.2f}")
        print(f"Tempo médio por batch: {stats['avg_inference_time']:.3f}s")
        print(f"Tempo médio por frame: {stats['time_per_frame']*1000:.1f}ms")
        print(f"Total de batches: {stats['total_batches']}")
        print("="*50)

    def detect_single(self, frame: np.ndarray):
        """Detecção em frame único (para fallback)"""
        start_time = time.time()
        
        result = self.model(frame, verbose=False)[0]
        
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.frame_count += 1
        
        return result
    
# Exemplo de uso otimizado:
if __name__ == "__main__":
    # Configurações para melhor performance
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/best.pt", help="Caminho do modelo")
    parser.add_argument("--video", default="data/video.mp4", help="Caminho do vídeo")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device para inferência")
    parser.add_argument("--batch_size", type=int, default=4, help="Tamanho do batch")
    parser.add_argument("--max_frames", type=int, default=100, help="Máximo de frames para processar")
    
    args = parser.parse_args()
    
    print(" Inicializando detector otimizado...")
    
    # Inicializa com batch menor para CPU
    batch_size = args.batch_size if args.device == "cuda" else min(args.batch_size, 4)
    
    detector = SignDetector(
        model_path=args.model,
        device=args.device,
        batch_size=batch_size,
        verbose=False  # Desativa logs para melhor performance
    )
    
    # Processa vídeo
    results = detector.process_video_batches(args.video, max_frames=args.max_frames)
    
    # Estatísticas
    detector.print_statistics()
    
    print(f"\n Processamento concluído!")
    print(f" Resultados: {len(results)} frames processados")