This repository presents our final project for CCS 248: Artificial Neural Networks, which involves building a deep learning model capable of classifying exercise form quality using wearable sensor data. The work applies a 1D-CNN/LSTM hybrid approach to process segmented time-series readings and distinguish correct biceps curl execution from common incorrect movement patterns, demonstrating practical applications of neural networks in human activity analysis.

## Project Overview

### Problem Statement
This project implements a **Long Short-Term Memory (LSTM) neural network** to classify exercise form quality using wearable sensor data. The goal is to distinguish between correct biceps curl execution (Class A) and common incorrect variations (Classes B-E) using multivariate time-series data from accelerometer, gyroscope, and magnetometer sensors.

### Dataset
- **Source**: Weight Lifting Exercises Dataset (UCI Machine Learning Repository)
- **Samples**: 4,024 segmented exercise repetitions
- **Features**: 52 sensor features (gyroscope, accelerometer, magnetometer readings from belt, arm, forearm, and dumbbell)
- **Classes**: 5 classes
  - **Class A**: Correct execution
  - **Class B**: Throwing elbows to the front
  - **Class C**: Lifting dumbbell only halfway
  - **Class D**: Lowering dumbbell only halfway
  - **Class E**: Throwing hips to the front
- **Format**: Segmented, multivariate time-series (each repetition is a sequence of sensor readings)

### Neural Network Architecture

**Model Type**: LSTM-based Time-Series Classifier

**Architecture Details**:
```
Input: [batch_size, sequence_length, num_features] (sequence of sensor readings)

LSTM Layer:
├── LSTM (input_size=num_features, hidden_size=128, num_layers=2, batch_first=True, dropout=0.5)

Fully Connected Layers:
├── Linear (128 → 64)
├── ReLU
├── Dropout (0.5)
└── Linear (64 → 5)

Output: [batch_size, 5] (class probabilities)
```

**Key Architecture Features**:
- **LSTM Layer**: Captures temporal dependencies in sensor sequences
- **Dropout**: Prevents overfitting
- **Fully Connected Classifier**: Maps LSTM output to exercise class labels

### Tools and Technologies

**Deep Learning Framework**:
- **PyTorch 2.9.1**: Neural network implementation, training, and inference
- **torch.nn**: Model architecture components (LSTM, Linear, etc.)
- **torch.optim**: Optimizers (Adam, SGD, RMSprop)
- **torch.utils.data**: DataLoader for efficient batch processing

**Data Processing**:
- **NumPy**: Numerical operations and array manipulation
- **Pandas**: Data loading, exploration, and preprocessing
- **scikit-learn**: 
  - `LabelEncoder`: Target label encoding
  - `train_test_split`: Data splitting (70% train, 15% validation, 15% test)
  - `classification_report`: Performance metrics
  - `confusion_matrix`: Error analysis

**Visualization**:
- **Matplotlib**: Training curves, loss/accuracy plots
- **Seaborn**: Confusion matrices, statistical visualizations

**Training Infrastructure**:
- **Learning Rate Scheduler**: ReduceLROnPlateau for adaptive learning rate
- **Early Stopping**: Based on validation accuracy
- **Loss Function**: CrossEntropyLoss for multi-class classification
- **Hardware**: CPU-based training (PyTorch CPU version)

**Development Environment**:
- **Jupyter Notebook**: Interactive development and documentation
- **Python 3.x**: Programming language
- **Device**: CPU (torch.device('cpu'))

### Training Configuration

**Data Split**:
- Training: 70% (sequences)
- Validation: 15% (sequences)
- Test: 15% (sequences)
- Stratified sampling to preserve class distribution

**Training Parameters**:
- **Epochs**: 50 (with early stopping)
- **Loss Function**: CrossEntropyLoss
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.5, patience=5)
- **Batch Processing**: Mini-batch gradient descent

**Preprocessing Pipeline**:
1. Feature selection (52 sensor features)
2. Missing value removal
3. Grouping sensor readings into sequences per exercise repetition
4. Label encoding (A→0, B→1, C→2, D→3, E→4)
5. Train/validation/test split
6. Sequence padding to uniform length
7. Tensor conversion for PyTorch

### Best Configuration

**Optimal Hyperparameters**:
- Model: LSTM
- Hidden Size: 128
- Layers: 2
- Dropout: 0.5
- Optimizer: Adam
- Learning Rate: 0.001
- Batch Size: 32

**Performance**:
- Test Accuracy: (to be determined)
- Validation Accuracy: (to be determined)
- Training Time: (to be determined)
