import cv2
import numpy as np

class VideoLoader:
    def __init__(self, batch_size=16):
        self.batch_size = batch_size
    
    def load_video(self, video_path):
        """Carrega vídeo e extrai frames"""
        print(f"Tentando abrir vídeo: {video_path}")
        
        # Verifica se o arquivo existe
        import os
        if not os.path.exists(video_path):
            print(f"ERRO: Arquivo não encontrado: {video_path}")
            print(f"Diretório atual: {os.getcwd()}")
            print(f"Conteúdo de data/: {os.listdir('data') if os.path.exists('data') else 'Diretório data não existe'}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"ERRO: Não foi possível abrir o vídeo: {video_path}")
            print(f"Codecs suportados? Tentando abrir com backend diferente...")
            
            # Tenta com backend diferente
            cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                print("ERRO: Falha mesmo com FFMPEG")
                return []
        
        frames = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frames.append(frame)
            frame_count += 1
            
            # Mostra progresso a cada 100 frames
            if frame_count % 100 == 0:
                print(f"  Carregados {frame_count} frames...")
        
        cap.release()
        
        print(f"Total de frames carregados: {frame_count}")
        print(f"Dimensão do primeiro frame: {frames[0].shape if frames else 'Nenhum frame'}")
        
        return frames
    
    def create_batches(self, frames):
        """Divide frames em batches"""
        if not frames:
            print("AVISO: Lista de frames vazia!")
            return []
        
        batches = []
        total_frames = len(frames)
        
        for i in range(0, total_frames, self.batch_size):
            batch_frames = frames[i:i + self.batch_size]
            
            # Verifica se todos os frames têm o mesmo tamanho
            shapes = [f.shape for f in batch_frames]
            if len(set(shapes)) > 1:
                print(f"AVISO: Frames com tamanhos diferentes no batch {i//self.batch_size}")
                # Redimensiona para o tamanho do primeiro frame
                target_shape = batch_frames[0].shape
                batch_frames = [cv2.resize(f, (target_shape[1], target_shape[0])) 
                              if f.shape != target_shape else f 
                              for f in batch_frames]
            
            # Converte para numpy array
            try:
                batch_array = np.stack(batch_frames)
                batches.append(batch_array)
            except ValueError as e:
                print(f"ERRO ao criar batch: {e}")
                print(f"Shapes dos frames: {shapes}")
                continue
        
        print(f"Criados {len(batches)} batches de tamanho {self.batch_size}")
        return batches