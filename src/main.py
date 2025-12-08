import argparse
import time
import cv2
import os
import sys
from tqdm import tqdm  
import numpy as np
import traceback

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from video_loader import VideoLoader
from detector import SignDetector
from visualizer import ResultVisualizer

def process_video_fast(input_path, output_path, model_path, max_frames=300, batch_size=8, 
                       start_frame=0, target_size=(640, 640), disable_ultralytics_logs=True):
    """
    Versão otimizada e robusta: processa segmento específico do vídeo
    
    Args:
        input_path: caminho do vídeo de entrada
        output_path: caminho do vídeo de saída
        model_path: caminho do modelo
        max_frames: quantidade máxima de frames a processar
        batch_size: tamanho do batch
        start_frame: frame inicial para começar o processamento
        target_size: tamanho padrão para redimensionamento (largura, altura)
        disable_ultralytics_logs: desativa logs verbosos do Ultralytics
    """
    print("="*60)
    print(" PROCESSADOR OTIMIZADO DE VÍDEO - DETECÇÃO DE SINAIS")
    print("="*60)
    
    # Verifica arquivos
    if not os.path.exists(input_path):
        print(f" Vídeo não encontrado: {input_path}")
        return False
    
    if not os.path.exists(model_path):
        print(f" Modelo não encontrado: {model_path}")
        return False
    
    print(f" Vídeo: {os.path.basename(input_path)}")
    print(f" Modelo: {os.path.basename(model_path)}")
    print(f" Frames: {start_frame} até {start_frame + max_frames - 1}")
    print(f" Batch size: {batch_size}")
    print(f" Tamanho alvo: {target_size[0]}x{target_size[1]}")
    
    # 1. Carrega segmento do vídeo
    print(f"\n[1/4]  Carregando frames...")
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        print(f" Não foi possível abrir o vídeo: {input_path}")
        return False
    
    # Obtém informações do vídeo
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"   • Frames total no vídeo: {total_frames}")
    print(f"   • FPS original: {original_fps:.2f}")
    print(f"   • Resolução original: {original_width}x{original_height}")
    
    # Ajusta max_frames se necessário
    if max_frames <= 0 or (start_frame + max_frames) > total_frames:
        max_frames = total_frames - start_frame
    
    if max_frames <= 0:
        print(f" Nenhum frame para processar")
        cap.release()
        return False
    
    # Posiciona no frame inicial
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    frame_indices = []  # Guarda índices originais dos frames
    
    print(f"   • Coletando {max_frames} frames...")
    pbar_frames = tqdm(total=max_frames, desc="Carregando frames", unit="frame")
    
    for i in range(max_frames):
        # Verifica se ainda há frames disponíveis
        current_pos = start_frame + i
        if current_pos >= total_frames:
            break
            
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        
        # Redimensiona para tamanho padrão
        if frame.shape[:2] != target_size[::-1]:  # cv2 usa (height, width)
            frame = cv2.resize(frame, target_size)
        
        frames.append(frame)
        frame_indices.append(current_pos)
        pbar_frames.update(1)
    
    pbar_frames.close()
    cap.release()
    
    if not frames:
        print(" Nenhum frame carregado")
        return False
    
    print(f"    Frames carregados: {len(frames)}")
    
    # 2. Cria batches de forma robusta
    print(f"\n[2/4]  Preparando batches...")
    batches = []
    batch_indices = []  # Guarda índices dos frames em cada batch
    
    total_batches = (len(frames) + batch_size - 1) // batch_size
    print(f"   • Total estimado de batches: {total_batches}")
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        current_indices = frame_indices[i:i + batch_size]
        
        if batch_frames:
            try:
                # Garante que todos os frames tenham o mesmo formato
                processed_batch = []
                for frame in batch_frames:
                    # Converte BGR para RGB (YOLO geralmente espera RGB)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # Garante o tipo de dados correto
                    frame_rgb = frame_rgb.astype(np.float32) / 255.0
                    processed_batch.append(frame_rgb)
                
                # Cria o batch
                batch_array = np.stack(processed_batch)
                batches.append(batch_array)
                batch_indices.append(current_indices)
                
                # Log do primeiro batch para debug
                if i == 0:
                    print(f"    Formato do batch: {batch_array.shape}")
                    print(f"    Tipo de dados: {batch_array.dtype}")
                    print(f"    Range de valores: [{batch_array.min():.3f}, {batch_array.max():.3f}]")
                
            except Exception as e:
                print(f"     Erro ao criar batch {len(batches)}: {str(e)[:100]}...")
                continue
    
    if not batches:
        print(" Nenhum batch criado")
        return False
    
    print(f"    Batches criados: {len(batches)}")
    
    # 3. Inicializa detector com configuração robusta
    print(f"\n[3/4] Inicializando detector...")
    
    try:
        # INICIALIZAÇÃO CORRIGIDA - use batch_size do argumento, não fixo
        detector = SignDetector(
            model_path=model_path,  # Note: model_path, não args.model
            device="cpu",           # ou "cuda" se tiver GPU
            batch_size=batch_size,  # Use o batch_size da função
            verbose=False           # Desativa logs para melhor performance
        )
        
    except Exception as e:
        print(f" Erro ao inicializar detector: {e}")
        print(f"   Stack trace:")
        traceback.print_exc()
        return False
    
    visualizer = ResultVisualizer()

    
    # 4. Processamento com tratamento robusto de erros
    print(f"\n[4/4] ⚡ Processando detecção...")
    processed_frames = []
    processed_indices = []
    processing_stats = {
        'total_batches': len(batches),
        'successful_batches': 0,
        'failed_batches': 0,
        'total_frames_processed': 0,
        'errors': []
    }
    
    start_time = time.time()
    
    # Configuração da barra de progresso
    pbar_batches = tqdm(zip(batches, batch_indices), 
                        total=len(batches),
                        desc="Processando batches",
                        unit="batch")
    
    for batch_idx, (batch, indices) in enumerate(pbar_batches):
        batch_success = False
        
        try:
            # Converte batch de RGB float32 0-1 para BGR uint8 0-255
            batch_bgr = []
            for frame_rgb in batch:
                frame_bgr = cv2.cvtColor((frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                batch_bgr.append(frame_bgr)

            results = detector.detect_batch(batch_bgr)
            
            # Processa cada frame do batch
            for j in range(len(batch)):
                try:
                    # Obtém frame original (não processado)
                    original_frame_idx = (batch_idx * batch_size) + j
                    if original_frame_idx < len(frames):
                        original_frame = frames[original_frame_idx].copy()
                        
                        # Desenha caixas se houver resultados
                        if j < len(results) and results[j] is not None:
                            frame_with_boxes = visualizer.draw_boxes(original_frame, results[j])
                        else:
                            frame_with_boxes = original_frame
                        
                        processed_frames.append(frame_with_boxes)
                        processed_indices.append(indices[j])
                
                except Exception as frame_error:
                    # Fallback para frame sem processamento
                    if original_frame_idx < len(frames):
                        processed_frames.append(frames[original_frame_idx].copy())
                        processed_indices.append(indices[j])
            
            processing_stats['successful_batches'] += 1
            processing_stats['total_frames_processed'] += len(batch)
            batch_success = True
            
        except Exception as batch_error:
            processing_stats['failed_batches'] += 1
            error_msg = f"Batch {batch_idx} (frames {indices[0]}-{indices[-1]}): {str(batch_error)[:200]}"
            processing_stats['errors'].append(error_msg)
            
            # Atualiza descrição da barra de progresso
            pbar_batches.set_postfix({
                'sucesso': processing_stats['successful_batches'],
                'falhas': processing_stats['failed_batches'],
                'erro': 'Batch'
            })
            
            print(f"\n     {error_msg}")
            
            # Fallback: processamento individual
            print(f"    Tentando processamento individual para batch {batch_idx}...")
            
            individual_success = 0
            for j, frame_rgb in enumerate(batch):
                try:
                    # Converte de volta para BGR para o modelo
                    frame_bgr = cv2.cvtColor((frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                    
                    # Processa individualmente com logging desativado
                    with HiddenPrints(disable_ultralytics_logs):
                        # Certifique-se que a classe SignDetector tem o método detect_single
                        result = detector.detect_single(frame_bgr)
                    
                    # Obtém frame original
                    original_frame_idx = (batch_idx * batch_size) + j
                    if original_frame_idx < len(frames):
                        original_frame = frames[original_frame_idx].copy()
                        
                        # Desenha caixas
                        if result is not None and hasattr(result, 'boxes'):
                            frame_with_boxes = visualizer.draw_boxes(original_frame, result)
                        else:
                            frame_with_boxes = original_frame
                        
                        processed_frames.append(frame_with_boxes)
                        processed_indices.append(indices[j])
                        individual_success += 1
                        
                except Exception as individual_error:
                    # Usa frame original como fallback
                    original_frame_idx = (batch_idx * batch_size) + j
                    if original_frame_idx < len(frames):
                        processed_frames.append(frames[original_frame_idx].copy())
                        processed_indices.append(indices[j])
            
            if individual_success > 0:
                print(f"    Recuperados {individual_success}/{len(batch)} frames")
                batch_success = True
        
        # Atualiza estatísticas na barra de progresso
        if batch_success:
            pbar_batches.set_postfix({
                'sucesso': processing_stats['successful_batches'],
                'falhas': processing_stats['failed_batches'],
                'status': 'OK'
            })
    
    pbar_batches.close()
    total_time = time.time() - start_time
    
    if not processed_frames:
        print(" Nenhum frame processado")
        return False
    
    # 5. Salvar vídeo
    print(f"\n Salvando vídeo ({len(processed_frames)} frames)...")
    
    # Determina FPS para saída
    output_fps = original_fps if original_fps > 0 else 20.0
    print(f"   • FPS de saída: {output_fps:.2f}")
    print(f"   • Resolução: {processed_frames[0].shape[1]}x{processed_frames[0].shape[0]}")
    
    # Tenta salvar o vídeo
    try:
        # Garante que o diretório de saída existe
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        # Usa o visualizador para salvar
        success = visualizer.save_video(processed_frames, output_path, fps=output_fps)
        
        if not success:
            # Fallback manual
            print("     Método do visualizador falhou, tentando salvar manualmente...")
            success = save_video_manual(processed_frames, output_path, fps=output_fps)
    
    except Exception as e:
        print(f"    Erro ao salvar vídeo: {e}")
        success = False
    
    # Recupera estatísticas do detector
    try:
        stats = detector.get_performance_stats()
    except:
        stats = {}
    
    # Cria arquivo de log com estatísticas
    log_file = output_path.replace('.mp4', '_stats.txt')
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("="*60 + "\n")
            f.write(" ESTATÍSTICAS DE PROCESSAMENTO\n")
            f.write("="*60 + "\n")
            f.write(f"Vídeo de entrada: {input_path}\n")
            f.write(f"Vídeo de saída: {output_path}\n")
            f.write(f"Modelo: {model_path}\n")
            f.write(f"Frames processados: {processed_indices[0]}-{processed_indices[-1]}\n")
            f.write(f"Total de frames: {len(processed_frames)}\n")
            f.write(f"Tempo total: {total_time:.2f}s\n")
            f.write(f"FPS médio: {len(processed_frames)/total_time:.2f}\n")
            f.write(f"Batches totais: {processing_stats['total_batches']}\n")
            f.write(f"Batches bem-sucedidos: {processing_stats['successful_batches']}\n")
            f.write(f"Batches com falha: {processing_stats['failed_batches']}\n")
            
            if processing_stats['errors']:
                f.write(f"\n--- Erros Encontrados ---\n")
                for error in processing_stats['errors']:
                    f.write(f"• {error}\n")
            
            if stats:
                f.write(f"\n--- Estatísticas do Detector ---\n")
                for key, value in stats.items():
                    f.write(f"{key}: {value}\n")
        
        print(f"     Log salvo: {log_file}")
    except Exception as e:
        print(f"     Não foi possível salvar log: {e}")
    
    if success:
        # 6. Exibe estatísticas
        print("\n" + "="*60)
        print(" ESTATÍSTICAS FINAIS:")
        print("="*60)
        print(f" Tempo total: {total_time:.2f}s")
        print(f" FPS médio: {len(processed_frames)/total_time:.2f}")
        print(f" Frames processados: {len(processed_frames)}")
        print(f" Batches: {processing_stats['total_batches']}")
        print(f" Batches bem-sucedidos: {processing_stats['successful_batches']}")
        print(f" Batches com falha: {processing_stats['failed_batches']}")
        
        if processing_stats['errors']:
            print(f"\n  Erros encontrados (primeiros 3):")
            for error in processing_stats['errors'][:3]:
                print(f"   • {error}")
            if len(processing_stats['errors']) > 3:
                print(f"   • ... e mais {len(processing_stats['errors']) - 3} erros")
        
        print(f"\n Arquivos gerados:")
        print(f"   • Vídeo: {output_path}")
        print(f"   • Log: {log_file}")
        print(f"\n Intervalo processado:")
        print(f"   • Frame inicial: {processed_indices[0]}")
        print(f"   • Frame final: {processed_indices[-1]}")
        print("="*60)
        
        return True
    else:
        print(" Falha ao salvar vídeo")
        
        # Tenta diagnosticar o problema
        if processed_frames:
            print(f"   • Frames disponíveis: {len(processed_frames)}")
            print(f"   • Formato do primeiro frame: {processed_frames[0].shape}")
            print(f"   • Tipo do primeiro frame: {processed_frames[0].dtype}")
            
            # Tenta salvar uma imagem de teste
            test_image = output_path.replace('.mp4', '_test.jpg')
            cv2.imwrite(test_image, processed_frames[0])
            print(f"   • Imagem de teste salva: {test_image}")
        
        return False


def save_video_manual(frames, output_path, fps=20.0):
    """Salva vídeo manualmente como fallback"""
    if not frames:
        return False
    
    try:
        height, width = frames[0].shape[:2]
        
        # Define o codec (MP4V para .mp4)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not out.isOpened():
            print(f"    Não foi possível criar VideoWriter para {output_path}")
            return False
        
        for frame in tqdm(frames, desc="Salvando frames", unit="frame"):
            # Garante que o frame está em BGR
            if frame.shape[-1] == 3:
                out.write(frame)
            else:
                # Converte se necessário
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
        
        out.release()
        print(f"    Vídeo salvo manualmente: {output_path}")
        return True
        
    except Exception as e:
        print(f"    Erro ao salvar vídeo manualmente: {e}")
        return False


class HiddenPrints:
    """Context manager para suprimir prints"""
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._original_stdout = None
    
    def __enter__(self):
        if self.enabled:
            self._original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            sys.stdout.close()
            sys.stdout = self._original_stdout


def process_video_segment(input_path, output_path, model_path, start_frame=0, 
                         end_frame=None, batch_size=8, target_size=(640, 640)):
    """
    Processa um segmento específico do vídeo
    """
    if end_frame is not None:
        max_frames = end_frame - start_frame + 1
    else:
        # Carrega vídeo para saber total de frames
        cap = cv2.VideoCapture(input_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        max_frames = total_frames - start_frame
    
    if max_frames <= 0:
        print(f" Intervalo inválido: start_frame={start_frame}, end_frame={end_frame}")
        return False
    
    return process_video_fast(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        max_frames=max_frames,
        batch_size=batch_size,
        start_frame=start_frame,
        target_size=target_size
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Processador otimizado de vídeo com YOLO - Detecção de Sinais",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos de uso:
  # Processa primeiros 300 frames
  python %(prog)s --video data/video.mp4 --max_frames 300
  
  # Processa segmento específico
  python %(prog)s --video data/video.mp4 --start_frame 1000 --end_frame 1300
  
  # Processa com batch maior
  python %(prog)s --video data/video.mp4 --batch_size 4
  
  # Processa vídeo inteiro (usar com cautela)
  python %(prog)s --video data/video.mp4 --max_frames 0
        """
    )
    
    parser.add_argument("--video", type=str, required=True,
                        help="Caminho do vídeo de entrada")
    
    parser.add_argument("--output", type=str, default="output_detected.mp4",
                        help="Caminho do vídeo de saída")
    
    parser.add_argument("--model", type=str, default="models/best.pt",
                        help="Caminho do modelo (.pt ou .onnx)")
    
    parser.add_argument("--max_frames", type=int, default=300,
                        help="Quantidade máxima de frames a processar (0 = todos)")
    
    parser.add_argument("--start_frame", type=int, default=0,
                        help="Frame inicial para começar processamento")
    
    parser.add_argument("--end_frame", type=int, default=None,
                        help="Frame final para parar processamento")
    
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Tamanho do batch (recomendado: 1-8 para CPU)")
    
    parser.add_argument("--target_width", type=int, default=640,
                        help="Largura alvo para redimensionamento")
    
    parser.add_argument("--target_height", type=int, default=640,
                        help="Altura alvo para redimensionamento")
    
    parser.add_argument("--show_logs", action="store_true",
                        help="Mostra logs detalhados do Ultralytics")
    
    args = parser.parse_args()
    
    print(" Configuração:")
    print(f"   • Batch size: {args.batch_size}")
    print(f"   • Max frames: {args.max_frames if args.max_frames > 0 else 'todos'}")
    print(f"   • Show logs: {'Sim' if args.show_logs else 'Não'}")
    print()
    
    # Processa baseado nos argumentos
    if args.end_frame is not None:
        # Modo segmento específico
        success = process_video_segment(
            input_path=args.video,
            output_path=args.output,
            model_path=args.model,
            start_frame=args.start_frame,
            end_frame=args.end_frame,
            batch_size=args.batch_size,
            target_size=(args.target_width, args.target_height)
        )
    else:
        # Modo máximo de frames
        success = process_video_fast(
            input_path=args.video,
            output_path=args.output,
            model_path=args.model,
            max_frames=args.max_frames,
            batch_size=args.batch_size,
            start_frame=args.start_frame,
            target_size=(args.target_width, args.target_height),
            disable_ultralytics_logs=not args.show_logs
        )
    
    if success:
        print("\n" + "="*60)
        print(" PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
        print("="*60)
        print(f" Vídeo resultante: {args.output}")
        print(f" Log detalhado: {args.output.replace('.mp4', '_stats.txt')}")
        print("="*60)
    else:
        print("\n" + "="*60)
        print(" PROCESSAMENTO FALHOU")
        print("="*60)
        sys.exit(1)