import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

def test_single_image():
    print("🧪 TESTE COM IMAGEM ÚNICA")
    print("="*50)
    
    # 1. Carrega modelo
    model = YOLO("models/yolov11n.pt")
    
    # 2. Carrega uma imagem do vídeo
    cap = cv2.VideoCapture("data/Video_transito.mp4")
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Não conseguiu ler frame do vídeo")
        return
    
    print(f"✅ Frame carregado: {frame.shape}")
    
    # 3. Faz detecção
    results = model(frame)
    
    # 4. Verifica resultados
    if results and results[0].boxes is not None:
        num_detections = len(results[0].boxes)
        print(f"✅ Detecções encontradas: {num_detections}")
        
        # Mostra cada detecção
        for i, box in enumerate(results[0].boxes):
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            print(f"  {i+1}: Classe {cls_id}, Conf: {conf:.2f}, Box: [{x1}, {y1}, {x2}, {y2}]")
        
        # 5. Plota resultado
        result_img = results[0].plot()  # Usa o plot do próprio YOLO
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Frame Original")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Com Detecções ({num_detections} boxes)")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Salva imagem de teste
        cv2.imwrite("debug_detection.jpg", result_img)
        print(f"💾 Imagem salva: debug_detection.jpg")
        
    else:
        print("❌ Nenhuma detecção encontrada!")
        print("   O modelo pode não estar treinado corretamente.")
        
        # Mostra frame original
        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        plt.title("Frame Original (sem detecções)")
        plt.axis('off')
        plt.show()

if __name__ == "__main__":
    test_single_image()