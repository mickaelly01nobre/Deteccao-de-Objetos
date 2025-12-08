# Pipeline de Detecção de Objetos

**Pré-requisitos do sistema**

Antes de executar o código, certifique-se de que as seguintes bibliotecas do sistema estejam instaladas:

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6
```

> **Observação:** Estas bibliotecas não estão listadas no `requirements.txt`, pois não são pacotes Python, mas dependências do sistema necessárias para o funcionamento do OpenCV e outros módulos gráficos.

**Execução do modelo**

 **Modelo padrão (.pt):**
 
```bash
python src/main.py --video data/Video_transito.mp4 --model models/best.pt --batch_size 8 --max_frames 300
``` 
**Modelo ONNX (.onnx):**

```bash
python src/main.py --video data/Video_transito.mp4 --model models/best.onnx --batch_size 8 --max_frames 300
``

**Modelo quantizado (INT8, .onnx):**

```bash
python src/main.py --video data/Video_transito.mp4 --model models/best_int8.onnx --batch_size 8 --max_frames 300
´´
´´
