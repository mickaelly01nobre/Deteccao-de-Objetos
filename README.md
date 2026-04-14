



# 🎯 Detecção de Objetos em Vídeo

## 📌 Sobre o Projeto

Este projeto tem como objetivo realizar a **detecção de objetos em vídeos** utilizando técnicas de **Visão Computacional** e modelos baseados em **YOLO (You Only Look Once)**.

A proposta é desenvolver um sistema capaz de:
- Ler um vídeo de entrada
- Processar seus frames
- Identificar objetos automaticamente
- Gerar um novo vídeo com as detecções destacadas

O projeto foi desenvolvido com foco em **eficiência, organização e robustez**, sendo adequado tanto para estudo quanto para aplicações reais.

---

## 🎯 Objetivo

O principal objetivo é construir um pipeline completo de detecção de objetos que:

- Seja **automatizado**
- Tenha **bom desempenho**
- Seja **tolerante a erros**
- Permita **processamento de grandes vídeos**

Além disso, o projeto pode ser utilizado como base para aplicações como:
- Sistemas de monitoramento
- Veículos autônomos
- Detecção de sinais de trânsito
- Análise de vídeo em tempo real

---

## 🧠 Como o Projeto Funciona

O funcionamento do sistema pode ser dividido em etapas:

### 1. 📥 Leitura do Vídeo
O vídeo é carregado utilizando a biblioteca OpenCV.

- Extração de informações:
  - FPS
  - Número de frames
  - Resolução

---

### 2. 🖼️ Processamento dos Frames
Cada frame do vídeo passa por um pré-processamento:

- Redimensionamento (para padronizar)
- Conversão de cor (BGR → RGB)
- Normalização dos valores

---

### 3. 📦 Organização em Batches
Os frames são agrupados em **batches** (lotes), o que melhora o desempenho da detecção.

- Reduz o custo computacional
- Aumenta a eficiência do modelo

---

### 4. 🧠 Detecção com YOLO
Os batches são enviados para um modelo YOLO que realiza a detecção dos objetos.

- O modelo identifica:
  - Objetos presentes
  - Localização (bounding boxes)
  - Confiança da detecção

---

### 5. 🎨 Visualização dos Resultados
Após a detecção:

- São desenhadas caixas (bounding boxes) nos objetos detectados
- O frame é atualizado com as informações visuais

---

### 6. 💾 Geração do Vídeo Final
Os frames processados são reunidos para gerar um novo vídeo contendo as detecções.

Além disso, é criado um arquivo de log com:
- Tempo de execução
- FPS médio
- Número de frames processados
- Estatísticas do processamento

---

## ⚙️ Tecnologias Utilizadas

O projeto foi desenvolvido utilizando as seguintes tecnologias:

### 🐍 Linguagem
- Python 3

### 📚 Bibliotecas
- **OpenCV** → processamento de vídeo e imagens  
- **NumPy** → manipulação de arrays  
- **tqdm** → barra de progresso  
- **Ultralytics (YOLO)** → modelo de detecção de objetos  

---



# Pipeline de Detecção de Objetos

**Pré-requisitos do sistema**

Antes de executar o código, certifique-se de que as seguintes bibliotecas do sistema estejam instaladas:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
```

> **Observação:** Estas bibliotecas não estão listadas no `requirements.txt`, pois não são pacotes Python, mas dependências do sistema necessárias para o funcionamento do OpenCV e outros módulos gráficos.

### Execução do modelo 

 **Modelo padrão (.pt):**
 
```bash
python src/main.py --video data/Video_transito.mp4 --model models/best.pt --batch_size 8 --max_frames 300
```

**Modelo ONNX (.onnx):**

```bash
python src/main.py --video data/Video_transito.mp4 --model models/best.onnx --batch_size 8 --max_frames 300
```

**Modelo quantizado (INT8, .onnx):**

```bash
python src/main.py --video data/Video_transito.mp4 --model models/best_int8.onnx --batch_size 8 --max_frames 300
```

# 🎯 Object Detection in Video

## 📌 About the Project

This project aims to perform **object detection in videos** using **Computer Vision** techniques and models based on **YOLO (You Only Look Once)**.

The goal is to develop a system capable of:
- Reading an input video
- Processing its frames
- Automatically identifying objects
- Generating a new video with highlighted detections

The project was developed with a focus on **efficiency, organization, and robustness**, making it suitable for both learning and real-world applications.

---

## 🎯 Objective

The main goal is to build a complete object detection pipeline that is:

- **Automated**
- High-performance
- **Error-tolerant**
- Capable of handling **large video processing**

Additionally, the project can be used as a foundation for applications such as:
- Monitoring systems
- Autonomous vehicles
- Traffic sign detection
- Real-time video analysis

---

## 🧠 How the Project Works

The system workflow can be divided into the following steps:

### 1. 📥 Video Input
The video is loaded using the OpenCV library.

- Extracted information:
  - FPS
  - Number of frames
  - Resolution

---

### 2. 🖼️ Frame Processing
Each video frame goes through preprocessing:

- Resizing (standardization)
- Color conversion (BGR → RGB)
- Value normalization

---

### 3. 📦 Batch Organization
Frames are grouped into **batches**, improving detection performance.

- Reduces computational cost
- Increases model efficiency

---

### 4. 🧠 Detection with YOLO
Batches are sent to a YOLO model for object detection.

- The model identifies:
  - Objects present
  - Locations (bounding boxes)
  - Detection confidence

---

### 5. 🎨 Result Visualization
After detection:

- Bounding boxes are drawn around detected objects
- Frames are updated with visual information

---

### 6. 💾 Final Video Generation
Processed frames are combined to generate a new video with detections.

Additionally, a log file is created containing:
- Execution time
- Average FPS
- Number of processed frames
- Processing statistics

---

## ⚙️ Technologies Used

The project was developed using the following technologies:

### 🐍 Language
- Python 3

### 📚 Libraries
- **OpenCV** → video and image processing  
- **NumPy** → array manipulation  
- **tqdm** → progress bar  
- **Ultralytics (YOLO)** → object detection model  

---

# Object Detection Pipeline

## System Requirements

Before running the code, make sure the following system libraries are installed:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
```


> **Note:** These libraries are not included in `requirements.txt because they are system-level dependencies required for OpenCV and other graphical modules.

 

### Model Execution

**Default model (.pt):**

 
```bash
python src/main.py --video data/Video_transito.mp4 --model models/best.pt --batch_size 8 --max_frames 300
```

**ONNX model (.onnx):**

```bash
python src/main.py --video data/Video_transito.mp4 --model models/best.onnx --batch_size 8 --max_frames 300
```

**Quantized model (INT8, .onnx):**

```bash
python src/main.py --video data/Video_transito.mp4 --model models/best_int8.onnx --batch_size 8 --max_frames 300
```
