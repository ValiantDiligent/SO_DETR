<h2 align="center">SO-DETR: Leveraging Dual-Domain Features and Knowledge Distillation for Small Object Detection</h2>


---

This is the official implementation of papers 
- [SO-DETR: Leveraging Dual-Domain Features and Knowledge Distillation for Small Object Detection]

## 🚀 Updates
- \[2025.01.17\] Release SO-DETR-R50, SO-DETR-R18, SO-DETR-EV2. Release the distillation code.

## 📍 Implementations
- 🔥 SO-DETR 
  - pytorch: [code&weight](./ultralytics)


### Experimental Results on the VisDrone-2019-DET Dataset

| **Model**            | **Backbone**         | **Input Size** | **Params (M)** | **GFLOPs** | **AP**  | **AP$_{50}$** |
|----------------------|---------------------|----------------|----------------|------------|---------|---------------|
| SO-DETR (Ours)       | EfficientFormerV2   | 640×640        | 12.1           | 33.3       | 28.2    | 46.7          |
| SO-DETR (Distilled)  | EfficientFormerV2   | 640×640        | 12.1           | 33.3       | 28.8    | 47.5          |
| SO-DETR (Ours)       | ResNet18            | 640×640        | 20.5           | 64.3       | 29.9    | 49.0          |
| SO-DETR (Ours)       | ResNet50            | 640×640        | 44.4           | 161.4      | 31.5    | 51.5          |

---

### Experimental Results on UAVVaste Dataset

| **Model**             | **Params (M)** | **GFLOPs** | **AP**  | **AP$_{50}$** |
|-----------------------|----------------|------------|---------|---------------|
| SO-DETR-R50 (Ours)    | 44.4           | 161.4      | 37.5    | 76.4          |
| SO-DETR-R18 (Ours)    | 20.5           | 64.3       | 35.1    | 72.1          |
| SO-DETR-EV2 (Ours)    | 12.1           | 33.3       | 33.7    | 70.6          |
| SO-DETR-EV2 (Distilled) | 12.1         | 33.3       | 36.9    | 73.6          |
