# MNIST CNN Classifier

This is my first project using Convolutional Neural Networks (CNNs).
I built a simple CNN model with PyTorch to classify handwritten digits from the MNIST dataset.

---

## Project Overview
- Dataset: [MNIST] (60k train, 10k test, 28x28 grayscale images).
- Model: Custom CNN with two convolutional layers and two fully connected layers.
- Optimizer: AdamW
- Loss Function: CrossEntropyLoss
- Device: Supports Apple MPS (Metal) or CPU.

---

## Model Architecture
- Conv2d(1 → 32, kernel=3, padding=same) → ReLU → MaxPool(2x2)  
- Conv2d(32 → 64, kernel=3, padding=same) → ReLU → MaxPool(2x2)  
- Flatten → Linear(64×7×7 → 28) → ReLU  
- Linear(28 → 10) (class scores)

---

## Training
Run the training script:
```bash
python MnistConv.py
```

The model trains for 10 epochs and prints the loss at each epoch.

---

## Example Output
```
Loss of Epoch 1 is 0.35
Loss of Epoch 2 is 0.12
...
```

---

## Future Improvements
- Add TensorBoard logging for visualization.
- Experiment with different optimizers (SGD, RMSprop).
- Add regularization (Dropout, BatchNorm).

---

This project is just the beginning of my deep learning journey with CNNs!
