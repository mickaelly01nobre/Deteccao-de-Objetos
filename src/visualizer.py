"""
 A classe ResultVisualizer é responsável por desenhar as detecções do modelo nos frames
 e salvar o resultado em vídeo. Ela mantém uma lista de nomes de classes (seguindo
 a ordem do data.yaml) e cores específicas para cada uma. Ao receber um frame e as
 detecções do YOLO, ela desenha as bounding boxes, escreve o nome da classe e a
 confiança, e devolve o frame anotado. Além disso, também permite salvar uma lista
 de frames processados como um vídeo final.
"""

import cv2
import numpy as np

class ResultVisualizer:
    def __init__(self):
        # Lista oficial das classes NA ORDEM DO data.yaml
        self.class_names = [
            'do_not_enter',
            'parking',
            'ped_zebra_cross',
            'red_light',
            'stop',
            'traffic_light',
            'warning'
        ]

        # Cores para cada classe
        self.colors = {
            'do_not_enter': (0, 0, 255),       # vermelho
            'parking': (0, 255, 0),            # verde
            'ped_zebra_cross': (255, 0, 0),    # azul
            'red_light': (0, 255, 255),        # amarelo
            'stop': (255, 0, 255),             # magenta
            'traffic_light': (255, 255, 0),    # ciano
            'warning': (255, 255, 255)         # branco
        }

    def draw_boxes(self, frame, detections):
        """Desenha bounding boxes no frame"""
        frame_copy = frame.copy()

        if detections and hasattr(detections, 'boxes'):
            for box in detections.boxes:

                # Coordenadas da box
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Classe e confiança
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # Nome da classe usando a ORDEM do data.yaml
                if cls_id < len(self.class_names):
                    class_name = self.class_names[cls_id]
                else:
                    class_name = f"Class {cls_id}"

                # Cor correspondente
                color = self.colors.get(class_name, (255, 255, 255))

                # Desenha caixa
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), color, 2)

                # Texto
                label = f"{class_name} {conf:.2f}"
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
