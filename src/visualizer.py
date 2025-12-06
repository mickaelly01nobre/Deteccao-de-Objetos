import cv2
import numpy as np

class ResultVisualizer:
    def __init__(self):
        # Cores para diferentes classes
        self.colors = {
            "stop_sign": (0, 0, 255),      # Vermelho
            "speed_limit": (0, 255, 0),    # Verde
            "yield": (255, 0, 0),          # Azul
            "traffic_light": (0, 255, 255), # Amarelo
            "pedestrian_crossing": (255, 0, 255), # Magenta
            "no_entry": (255, 255, 0)      # Ciano
        }
    
    def draw_boxes(self, frame, detections):
        """
        Desenha bounding boxes no frame
        Args:
            frame: imagem original
            detections: objeto Results do YOLO
        Returns:
            Frame com boxes desenhadas
        """
        frame_copy = frame.copy()
        
        if detections and hasattr(detections, 'boxes'):
            for box in detections.boxes:
                # Coordenadas da box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Classe e confiança
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                
                # Nome da classe (ajuste baseado nas suas classes)
                class_names = ["stop_sign", "speed_limit", "yield", 
                              "traffic_light", "pedestrian_crossing", "no_entry"]
                class_name = class_names[cls_id] if cls_id < len(class_names) else f"Class {cls_id}"
                
                # Cor da classe
                color = self.colors.get(class_name, (255, 255, 255))
                
                # Desenha retângulo
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)
                
                # Texto com classe e confiança
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(frame_copy, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame_copy
    
    def save_video(self, frames, output_path, fps=30):
        """Salva lista de frames como vídeo"""
        if not frames:
            return
        
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        for frame in frames:
            out.write(frame)
        
        out.release()