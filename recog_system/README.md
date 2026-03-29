# 🖐️ Sistema de Reconhecimento & Laboratório de Visão Computacional

![Versão do Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-00C7B7?style=for-the-badge&logo=google&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

Este repositório contém um sistema completo para reconhecimento de gestos em tempo real utilizando **MediaPipe** e **Scikit-Learn**, além de uma coleção de notebooks exploratórios para diversas tarefas de Visão Computacional.

[⬅️ Voltar para a Raiz do Projeto](../README.md)

---

## 🚀 Sobre o Projeto

O coração deste projeto é um pipeline de ponta a ponta que permite coletar dados das mãos, treinar um classificador customizado e realizar inferência em tempo real via webcam. Ao contrário de modelos pré-treinados estáticos, este sistema permite que os usuários criem seus próprios gestos e treinem o modelo para reconhecê-los com alta precisão.

---

## 🛠️ Construído Com

- **Python**: Linguagem principal.
- **OpenCV**: Captura e processamento de vídeo.
- **MediaPipe**: Extração robusta de marcos (landmarks) das mãos.
- **Scikit-Learn**: Treinamento de modelos de Machine Learning (Random Forest).
- **Pandas/NumPy**: Manipulação e processamento de dados.
- **uv**: Gerenciamento de dependências e ambiente.

---

## 📽️ Pipeline de Reconhecimento de Gestos

O fluxo de trabalho está dividido em etapas principais dentro de notebooks interativos:

### 1. Coleta de Dados (`data_collection.ipynb`)
Use este notebook para criar seu dataset customizado. Ele captura as coordenadas (x, y, z) dos 21 marcos da mão através da webcam.
- **Funcionalidade:** Grava marcos extraídos via MediaPipe diretamente em um arquivo CSV.

### 2. Treinamento do Modelo (`train_gesture_model.ipynb`)
Processa o arquivo `hand_landmarks_data.csv` gerado e treina um classificador `RandomForest`.
- **Saída:** Gera `gesture_model.pkl` e `label_encoder.pkl`.
- Realiza avaliação de métricas (Matriz de Confusão, Precision/Recall).

### 3. Reconhecimento em Tempo Real
A lógica de inferência em tempo real é explorada no notebook `mediapipe_object_detection.ipynb` (adaptado para gestos) e serve de base para a aplicação web principal.

---

## 📓 Notebooks Exploratórios

Além do sistema de gestos, o repositório inclui implementações de ponta para outras subáreas da Visão Computacional:

- **`timm_classification_test.ipynb`**: Classificação de imagens usando a biblioteca `timm` (MobileNetV3).
- **`yolos_detection_test.ipynb`**: Detecção de objetos utilizando o modelo YOLO-S do Hugging Face.
- **`clipseg_segmentation.ipynb`**: Segmentação de imagem baseada em texto com CLIPSeg.
- **`gemini_vision_test.ipynb`**: Integração com a API do Google Gemini para análise multimodal de imagens.
- **`mediapipe_object_detection.ipynb`**: Implementação base para detecção de marcas e objetos via MediaPipe.

---

## ⚙️ Instalação & Configuração

### 1. Pré-requisitos
Este projeto utiliza o [uv](https://github.com/astral-sh/uv) para gerenciamento de pacotes.

### 2. Instalação
Navegue até a pasta e sincronize as dependências:
1. Clone o repositório.
2. Sincronize as dependências:
   ```bash
   cd recog_system
   uv sync
   ```

### 3. Download dos Modelos
Para garantir que o sistema funcione, baixe os modelos pré-treinados do MediaPipe e coloque-os na raiz do projeto:

- **[gesture_recognizer.task](https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task)**: Essencial para extração de marcos das mãos.
- **[efficientdet_lite0.tflite](https://storage.googleapis.com/mediapipe-models/object_detector/efficientdet_lite0/float16/1/efficientdet_lite0.tflite)**: Usado para detecção de objetos em notebooks auxiliares.

---

## 📂 Estrutura do Projeto

```text
recog_system/
├── data_collection.ipynb    # Ferramenta de criação de dataset
├── train_gesture_model.ipynb # Notebook de treinamento de ML
├── timm_classification_test.ipynb # Exploração MobileNetV3
├── yolos_detection_test.ipynb # Exploração YOLO-S
├── clipseg_segmentation.ipynb # Exploração CLIPSeg
├── gemini_vision_test.ipynb # Exploração Google Gemini
├── mediapipe_object_detection.ipynb # Detecção MediaPipe
├── pyproject.toml          # Metadados e dependências
└── README.md               # Você está aqui!
```

---

*Feito durante a NLW @ Rocketseat*
