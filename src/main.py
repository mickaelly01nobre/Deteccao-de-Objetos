import argparse
import time
import cv2
import os
import sys
from tqdm import tqdm  # Para progress bar

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_loader import VideoLoader
from detector import SignDetector
from visualizer import ResultVisualizer

def process_video_fast(input_path, output_path, model_path, max_frames=200, batch_size=8):
    """
    Versão otimizada: processa apenas os primeiros N frames
    """
    print("="*50)
    print("EXECUÇÃO OTIMIZADA (apenas primeiros frames)")
    print("="*50)
    
    # Verifica arquivos
    if not os.path.exists(input_path):
        print(f" Vídeo não encontrado: {input_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f" Modelo não encontrado: {model_path}")
        return False
    
    # 1. Carrega vídeo (apenas primeiros frames)
    print(f"\n[1/4] Carregando primeiros {max_frames} frames...")
    loader = VideoLoader(batch_size=batch_size)
    
    # Modifica o video_loader para carregar menos frames
    cap = cv2.VideoCapture(input_path)
    frames = []
    for i in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    
    if not frames:
        print(" Nenhum frame carregado")
        return False
    
    print(f"✓ Frames carregados: {len(frames)}")
    
    # 2. Cria batches
    print(f"\n[2/4] Criando batches...")
    batches = []
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        if len(batch_frames) == batch_size or i == 0:  # Pelo menos um batch completo
            try:
                batch_array = np.stack(batch_frames)
                batches.append(batch_array)
            except:
                continue
    
    if not batches:
        print(" Nenhum batch criado")
        return False
    
    print(f"✓ Batches criados: {len(batches)}")
    
    # 3. Inicializa detector (CPU para menos memória)
    print(f"\n[3/4] Inicializando detector...")
    detector = SignDetector(model_path, device="cpu", batch_size=batch_size)
    visualizer = ResultVisualizer()
    
    # 4. Processa com progress bar
    print(f"\n[4/4] Processando detecção...")
    processed_frames = []
    start_time = time.time()
    
    for i, batch in enumerate(tqdm(batches, desc="Processando batches")):
        try:
            results = detector.detect_batch(batch)
            
            # Processa resultados
            batch_start_idx = i * batch_size
            for j in range(min(batch.shape[0], len(results))):
                frame_idx = batch_start_idx + j
                if frame_idx < len(frames):
                    frame_result = results[j] if j < len(results) else None
                    frame_with_boxes = visualizer.draw_boxes(frames[frame_idx], frame_result)
                    processed_frames.append(frame_with_boxes)
        except Exception as e:
            print(f"  Erro no batch {i}: {e}")
            continue
    
    total_time = time.time() - start_time
    
    # 5. Salva vídeo reduzido
    if processed_frames:
        print(f"\n Salvando vídeo ({len(processed_frames)} frames)...")
        visualizer.save_video(processed_frames, output_path, fps=20)  # FPS menor para arquivo menor
        
        # Estatísticas
        stats = detector.get_performance_stats()
        
        print("\n" + "="*50)
        print("📊 ESTATÍSTICAS:")
        print("="*50)
        print(f"Tempo total: {total_time:.1f}s")
        print(f"FPS: {len(processed_frames)/total_time:.1f}")
        print(f"Frames processados: {len(processed_frames)}")
        print(f"Tamanho médio por batch: {stats['avg_inference_time']:.3f}s")
        print(f"Arquivo salvo: {output_path}")
        print("="*50)
        
        return True
    else:
        print(" Nenhum frame processado")
        return False

if __name__ == "__main__":
    import numpy as np
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, default="data/Video_transito.mp4")
    parser.add_argument("--output", type=str, default="output_fast.mp4")
    parser.add_argument("--model", type=str, default="models/yolov11n.pt")
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    
    args = parser.parse_args()
    
    success = process_video_fast(
        args.video, 
        args.output, 
        args.model, 
        max_frames=args.max_frames,
        batch_size=args.batch_size
    )
    
    if success:
        print("\n PROCESSO CONCLUÍDO COM SUCESSO!")
        print(f" Assista o resultado: {args.output}")
    else:
        print("\n PROCESSO FALHOU")