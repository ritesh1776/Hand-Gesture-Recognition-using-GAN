# 📌 Hand Gesture Recognition using GANs and CNNs

## 📖 Overview  
This project presents a hybrid deep learning approach for hand gesture recognition by integrating **Generative Adversarial Networks (GANs)** with **Convolutional Neural Networks (CNNs)**. Traditional models struggle with limited data and performance under varying conditions. To overcome these issues, GANs are employed to generate synthetic data, enhancing model generalization and accuracy.

---

## 🎯 Objectives  
- Improve gesture classification accuracy through data augmentation using GANs.  
- Design a robust CNN architecture with advanced feature extraction capabilities.  
- Reduce reliance on large labeled datasets while ensuring adaptability in real-world scenarios.

---

## 🧠 Approach  
1. **Synthetic Data Generation**: GANs are used to generate diverse hand gesture images, enriching the training dataset.  
2. **CNN Training**: The augmented dataset is used to train a CNN for accurate gesture classification.  
3. **Model Optimization**: A custom CNN architecture combining parallel and sequential layers is implemented for better performance.  
4. **Evaluation**: Compared the performance of three models:
   - Baseline CNN  
   - CNN with GAN-augmented data  
   - Modified CNN (parallel + sequential architecture)

---

## 🧾 Dataset  
- **Source**: Massey University Hand Gesture Dataset  
- **Classes**: 36 gesture categories  
- **Format**: Pre-split into training and testing sets  
- **Image Type**: Grayscale, resized to 64×64×1

---

## 🧩 Model Specifications

### 🔹 Input Layer
- **Input shape**: 64×64×1 (grayscale images)

### 🔹 Convolutional Layers
- **Total**: 8  
  - 4 in **parallel branches**  
  - 4 in **sequential layers**  
- Each followed by **Batch Normalization** to improve training stability

### 🔹 Pooling Layers
- **Total**: 6  
  - 4 in parallel branches  
  - 2 in sequential layers  
- All use **MaxPooling2D** to reduce spatial dimensions

### 🔹 Dropout Layers
- **Count**: 3  
- Added to reduce overfitting

### 🔹 Skip Connections
- 4 **parallel branches** concatenated to preserve multi-level feature representations

### 🔹 Global Average Pooling (GAP)
- Aggregates spatial information before passing to dense layers

### 🔹 Dense Layers
- **Total**: 3 fully connected layers  
- Final layer uses **SoftMax activation** for multi-class classification

### ✅ Summary
- **Parallel Layers**: 4 convolutional, 4 batch normalization, 4 pooling  
- **Sequential Layers**: 4 convolutional, 4 batch normalization, 2 pooling, 3 dense  
- **Design Focus**: Efficient feature extraction, regularization, and high classification performance

---

## 📊 Results  
- **CNN-only (baseline)**: ~82% accuracy  
- **CNN with GAN augmentation**: ~86% accuracy  
- **Modified CNN (parallel + sequential)**: **~89.5% accuracy**  
- Noticeable improvement in robustness across lighting, motion, and background variations

---

## 💻 Technologies Used  
- Python  
- TensorFlow / Keras  
- NumPy, OpenCV  
- Matplotlib (visualization)

---


## 📈 Future Improvements  
- Integrate real-time gesture recognition using webcam input  
- Extend the system for dynamic gestures (video sequences)  
- Deploy on edge devices for real-time HCI applications

---

## 🙌 Acknowledgements  
- Massey University for providing the dataset  
- Open-source contributors and deep learning research community

© 2025 Ritesh.
