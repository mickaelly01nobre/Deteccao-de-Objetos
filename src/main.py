import argparse
import time
import cv2
import os
import sys

# Adiciona o diretório src ao path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_loader import VideoLoader
from detector import SignDetector
from visualizer import ResultVisualizer

def main():
    parser = argparse.ArgumentParser(description="Detector de Placas de Trânsito")
    parser.add_argument("--video", type=str, required=True, help="Caminho do vídeo de entrada")
    parser.add_argument("--output", type=str, default="output.mp4", help="Caminho do vídeo de saída")
    parser.add_argument("--model", type=str, default="models/yolov11n.pt", help="Caminho do modelo")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamanho do batch")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Verifica se o modelo existe
    if not os.path.exists(args.model):
        print(f"ERRO: Modelo não encontrado: {args.model}")
        print(f"Modelos disponíveis em models/: {os.listdir('models') if os.path.exists('models') else 'Diretório models não existe'}")
        return
    
    # Inicializa componentes
    print("="*50)
    print("INICIALIZANDO DETECTOR DE PLACAS")
    print("="*50)
    
    loader = VideoLoader(batch_size=args.batch_size)
    detector = SignDetector(args.model, device=args.device, batch_size=args.batch_size)  # Passe batch_size aqui
    visualizer = ResultVisualizer()
    
    # Carrega vídeo
    print(f"\n[1/4] Carregando vídeo: {args.video}")
    frames = loader.load_video(args.video)
    
    if not frames:
        print("ERRO: Nenhum frame foi carregado. Verifique:")
        print("  1. Se o caminho do vídeo está correto")
        print("  2. Se o vídeo tem um codec suportado (MP4 com H.264)")
        print("  3. Se o OpenCV está instalado corretamente")
        print("\nDica: Tente converter o vídeo com:")
        print("  ffmpeg -i seu_video.avi -c:v libx264 -c:a aac output.mp4")
        return
    
    print(f"✓ Total de frames: {len(frames)}")
    
    # Cria batches
    print(f"\n[2/4] Criando batches (tamanho: {args.batch_size})...")
    batches = loader.create_batches(frames)
    
    if not batches:
        print("ERRO: Nenhum batch foi criado")
        return
    
    print(f"✓ Total de batches: {len(batches)}")
    
    # Processa cada batch
    print(f"\n[3/4] Processando detecção...")
    processed_frames = []
    start_total = time.time()
    
    for i, batch in enumerate(batches):
        print(f"  Batch {i+1}/{len(batches)}...")
        
        # Faz inferência no batch
        results = detector.detect_batch(batch)
        
        # Processa resultados de cada frame no batch
        batch_start_idx = i * args.batch_size
        
        for j in range(batch.shape[0]):
            frame_idx = batch_start_idx + j
            
            if frame_idx < len(frames):
                # Pega o resultado correspondente
                frame_result = results[j] if j < len(results) else None
                
                # Desenha boxes no frame
                frame_with_boxes = visualizer.draw_boxes(frames[frame_idx], frame_result)
                processed_frames.append(frame_with_boxes)
    
    total_time = time.time() - start_total
    
    if not processed_frames:
        print("ERRO: Nenhum frame processado")
        return
    
    # Salva vídeo processado
    print(f"\n[4/4] Salvando vídeo processado: {args.output}")
    visualizer.save_video(processed_frames, args.output, fps=30)
    print(f"✓ Vídeo salvo com sucesso!")
    
    # Exibe estatísticas
    stats = detector.get_performance_stats()
    print("\n" + "="*50)
    print("ESTATÍSTICAS DE PERFORMANCE:")
    print("="*50)
    print(f"Tempo total de processamento: {total_time:.2f}s")
    print(f"Tempo médio por batch: {stats['avg_inference_time']:.3f}s")
    print(f"FPS por batch: {stats['fps_per_batch']:.1f}")
    print(f"FPS total: {len(frames)/total_time:.1f}" if total_time > 0 else "FPS total: N/A")
    print(f"Total de frames processados: {len(processed_frames)}")
    print(f"Total de batches: {stats['total_batches']}")
    print("="*50)

if __name__ == "__main__":
    main()