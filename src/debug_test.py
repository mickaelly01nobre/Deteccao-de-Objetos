# test_videoloader.py

from video_loader import VideoLoader

def main():
    video_path = "data/Video_trasito.mp4"   # coloque o caminho do seu vídeo

    # Cria o carregador de vídeo com batch de 16 frames
    loader = VideoLoader(batch_size=16)

    print("\n=== Teste: Carregando vídeo ===")
    frames = loader.load_video(video_path)

    if len(frames) == 0:
        print("❌ ERRO: Nenhum frame foi carregado.")
        return

    print(f"✔️ Vídeo carregado com {len(frames)} frames.")

    print("\n=== Teste: Criando batches ===")
    batches = loader.create_batches(frames)

    if len(batches) == 0:
        print("❌ ERRO: Nenhum batch foi criado.")
        return

    print(f"✔️ {len(batches)} batches criados.")
    print(f"✔️ Cada batch tem forma: {batches[0].shape} (deve ser algo como (16, H, W, 3))")

    print("\n=== Teste finalizado com sucesso ===")

if __name__ == "__main__":
    main()
