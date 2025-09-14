# IoU-Based Overlapping Metric and Aggregated Loss Function for Morphological-Linear Neural Networks

## Description
This project implements a novel neural network optimization framework that uses Dendral Neurons and the Overlapping Index over Union (OIoU) metric to automatically reduce network complexity while maintaining classification performance. The system employs two optimization algorithms that progressively reduce the number of neurons based on overlapping measurements between hypercube-based neuron representations, achieving 20-50% parameter reduction with minimal accuracy loss.

## Dataset Information

### Synthetic Datasets
- **Dataset A**: Binary classification problem with 2D features, 1000 training samples, 1000 test samples
- **Dataset B**: 3-class classification problem with 2D features
- **XOR Dataset**: Non-linearly separable XOR problem with Gaussian noise, 2D features
- **Spiral Dataset**: Configurable spiral patterns with:
  - 2 or 3 classes
  - 1, 2, or 5 loops/spirals
  - Balanced class distribution

### Real-World Datasets
- **MNIST**: 60,000 training images, 10,000 test images, 784 features (28x28 pixels), 10 classes
- **Letter Recognition**: 20,000 samples, 16 numerical features, 26 classes (A-Z)
- **IMDB Movie Reviews**: 50,000 reviews for sentiment analysis, Word2Vec embeddings (100 dimensions)
- **Artificial Characters**: Murphy's artificial character dataset with 36 features, 10 character classes

## Code Information

### Core Components
- **DendralNeuron.py**: Custom Keras layer implementing dendral neurons using dual weight matrices (Wmin, Wmax) to create hypercubes in feature space
- **Tensor_IoU.py**: TensorFlow implementation of OIoU metric calculation with JIT compilation for performance
- **IoU_Loss.py**: Hybrid loss function combining classification loss with OIoU penalty term
- **ModelUtils.py**: Contains Algorithm 1 (architecture reduction) and Algorithm 2 (OIoU loss optimization)
- **CustomCallback.py**: Keras callback for real-time OIoU metric monitoring during training

### Main Scripts
- **MLNN_IoU.py**: Main execution script for running experiments
- **GPU_Test.py**: GPU availability and TensorFlow environment verification
- **BuildModel.py**: Model construction and training utilities
- **DatasourceUtils.py**: Dataset loading and preprocessing functions
- **PlotUtils.py**: Visualization tools for metrics and decision boundaries

## Usage Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/your-username/OIoU-Metrics.git
cd OIoU-Metrics

# Install required packages
pip install -r requirements.txt
```

### 2. Running Experiments
```python
# Basic execution with default Spiral dataset
python MLNN_IoU.py

# To use different datasets, edit MLNN_IoU.py:
def main():
    mu.enviroment()
    mu.reproducibility()
    
    # Uncomment desired dataset:
    mu.classify_DataSet_A()        # For Dataset A
    # mu.classify_DataSet_XOR()    # For XOR Dataset
    # mu.classify_DataSet_Spiral()  # For Spiral Dataset
```

### 3. Custom Configuration
```python
# Configure hyperparameters in ModelUtils.py
mu.classify_DataSet_Spiral(
    dendral_neurons=402,     # Initial architecture size
    lr=0.00992,             # Learning rate
    activation='tanh',      # Activation function
    v_verbose=True          # Enable verbose output
)
```

### 4. Loading Custom Datasets
```python
# In DatasourceUtils.py, implement:
def load_custom_dataset(normalize=False, to_categorical=True):
    # Load your data
    # Preprocess features
    # Return: train_dataset, test_dataset, (P, T), (Ptest, Ttest), [input_dim, num_classes, 'DATA']
```

## Requirements

### Python Version
- 
Python 3.9.18 

### Dependencies
```
Numpy  1.26.0
Tensor Flow 2.10.1
Keras 2.10.0
GPU 
Eager execution: True 
NLTK (for text processing)
Gensim (for Word2Vec)
```

### Hardware Requirements
- GPU recommended for large datasets (MNIST, IMDB)
- Minimum 8GB RAM for text processing tasks
- CUDA-compatible GPU for TensorFlow GPU acceleration (optional)

## Methodology

### Algorithm 1: Architecture Reduction
1. **Initial Training**: Train model with full architecture using standard cross-entropy loss
2. **OIoU Calculation**: Compute overlapping metric between all neuron pairs
3. **Reduction Rate Determination**: Calculate reduction percentage based on OIoU value
4. **Iterative Reduction**: Perform up to 10 iterations:
   - Reduce neurons by calculated percentage
   - Retrain model
   - Keep configuration if accuracy is maintained
5. **Output**: Optimized architecture with reduced parameters

### Algorithm 2: OIoU Loss Optimization
1. **Architecture Input**: Use optimized architecture from Algorithm 1
2. **Loss Function Modification**: Add OIoU term to classification loss:
   ```
   Total Loss = Classification Loss + λ * OIoU
   ```
3. **Hyperparameter Search**: Test λ values: [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
4. **Selection**: Choose λ that maximizes validation accuracy
5. **Output**: Final optimized model with best performance

### OIoU Metric Computation
1. **Hypercube Construction**: Each dendral neuron defines a hypercube using Wmin and Wmax
2. **Intersection Calculation**: Compute intersection volume between all neuron pairs
3. **Union Calculation**: Compute union volume for each pair
4. **OIoU Score**: Average IoU across all unique pairs, expressed as percentage

## Citations

If you use this code in your research, please cite:

```bibtex
@software{oiou_metrics_2025,
  title = {IoU-Based Overlapping Metric and Aggregated Loss Function for Morphological-Linear Neural Networks},
  author = {[Gerardo Hernández-Hernández]},
  year = {2025},
  url = {https://github.com/your-username/OIoU-Metrics},
  note = {Software available at: https://github.com/your-username/OIoU-Metrics}
}
```

### Related Publications
- [Add any related papers or references here]

## License & Contribution Guidelines

### License
This project is licensed under the MIT License. See `LICENSE` file for details.

### Contributing
We welcome contributions! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/new-feature`)
3. **Commit** changes with clear messages (`git commit -m 'Add new feature'`)
4. **Push** to your fork (`git push origin feature/new-feature`)
5. **Submit** a Pull Request with detailed description

### Code Style
- Follow PEP 8 for Python code
- Add docstrings to all functions
- Include type hints where applicable
- Write unit tests for new features

### Reporting Issues
Please use the GitHub Issues tracker to report bugs or request features. Include:
- Python version and environment details
- Complete error messages
- Minimal reproducible example
- Expected vs actual behavior


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////
# IoU-Based Overlapping Metric and Aggregated Loss Function for Morphological-Linear Neural Networks

## ABSTRACT
<center>
This paper proposes two new metrics based on the concept of Intersection over Union (IoU). The first
one is the Overlapping Intersection over Union (OIoU), which provides an accurate measure of the
overlap level among morphological neurons with dendritic processing that make up the first layer of
Morphological-Linear Neural Networks (MLNNs). This metric enables a reduction of more than 30% in
the size of the first layer of MLNNs. The second metric is the added cost function OIoU-Loss, this metric
serves as an additional loss function integrated into the original cost function. Its objective is to minimize
the overlap allowed between morphological neurons with dendritic processing, thereby enabling effective
pattern classification and, at the same time, the separation of the hyperboxes in MLNNs.
</center>

## 📋 Description
<center> 
This project implements an innovative neural network architecture utilizing **Dendral Neurons** and the **OIoU (Overlapping Index over Union)** metric for automatic neural architecture optimization. The system significantly reduces the number of parameters while maintaining or improving classification performance.
</center>
## 🎯 Key Features

### Novel Architecture
- **Dendral Neurons**: Custom layer implementation using hypercubes in feature space through dual weight matrices (Wmin and Wmax)
- **OIoU Metric**: Overlapping measurement between hypercubes to evaluate neuron redundancy
- **Automatic Optimization**: Algorithms that progressively reduce network size based on overlapping metrics

### Technical Components
- Hybrid loss function combining classification loss with overlapping penalty
- Custom callbacks for real-time OIoU metric monitoring
- Support for binary and multi-class classification
- Optimized implementation with TensorFlow 2.x and JIT-compiled operations

## 🚀 Installation

### Prerequisites
```bash
Python 3.9.18 
Numpy  1.26.0
Tensor Flow 2.10.1
Keras 2.10.0
GPU 
Eager execution: True 
NLTK (for text processing)
Gensim (for Word2Vec)
```

### Environment Setup
```bash
# Clone repository
git clone https://github.com/your-username/OIoU-Metrics.git
cd OIoU-Metrics
```

## 📁 Project Structure

```
OIoU_Metrics/
│
├── MLNN_IoU.py                 # Main execution script
├── ModelUtils.py               # Utilities and optimization algorithms
├── GPU_Test.py                 # GPU availability verification
│
├── model/
│   ├── BuildModel.py           # Model construction and training
│   └── DendralNeuron.py        # Dendral Neuron layer implementation
│
├── iou/
│   └── Tensor_IoU.py           # OIoU metrics calculation with TensorFlow
│
├── loss/
│   └── IoU_Loss.py             # Custom loss function with OIoU
│
├── callback/
│   └── CustomCallback.py       # Callbacks for metric monitoring
│
├── Datasource/
│   └── DatasourceUtils.py      # Dataset loading and preprocessing
│
└── Plot/
    └── PlotUtils.py            # Results and metrics visualization
```

## 💻 Usage

### Basic Execution

```python
# Run experiment with Spiral Dataset
python MLNN_IoU.py
```

### Dataset Selection

In `MLNN_IoU.py`, uncomment the corresponding dataset line:

```python
def main():
    mu.enviroment()
    mu.reproducibility()
    
    # Select one of the following datasets:
    
    # mu.classify_DataSet_A()        # Dataset A (2 classes, 2D)
    # mu.classify_DataSet_XOR()      # XOR Dataset (2 classes, 2D)
    mu.classify_DataSet_Spiral()     # Spiral Dataset (configurable)
```

### Parameter Configuration

Each classification function accepts customizable parameters:

```python
mu.classify_DataSet_Spiral(
    dendral_neurons=402,              # Initial number of dendral neurons
    lr=0.009922894436983054,          # Learning rate
    activation='tanh',                # Activation function
    v_verbose=False                   # Training verbosity
)
```

## 🔬 Optimization Algorithms

### Algorithm 1: Architecture Reduction
Iteratively reduces neuron count based on OIoU metric:

1. Trains initial model with full architecture
2. Calculates reduction rate based on OIoU
3. Progressively reduces neurons (up to 10 iterations)
4. Maintains architecture that preserves classification performance

**Key Parameters:**
- `num_of_customization_trials`: Number of reduction iterations
- `percentaje_of_custom_trials`: Reduction factor (0.25 or 0.75 based on initial OIoU)

### Algorithm 2: OIoU Loss Optimization
Retrains optimized model using different OIoU component weights:

```python
LR_Iou = [0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]
```

Automatically selects weight that maximizes validation accuracy.

## 📊 Supported Datasets

### Synthetic Datasets
- **Dataset A**: Binary classification, 2D
- **Dataset B**: 3-class classification, 2D
- **XOR Dataset**: XOR problem with Gaussian distribution
- **Spiral Dataset**: Configurable spirals (2-3 classes, 1-5 loops)

### Real-World Datasets
- **MNIST**: Handwritten digits (28x28 → 784 features)
- **Letter Recognition**: Letter recognition (16 features)
- **IMDB**: Sentiment analysis with Word2Vec (100 dimensions)
- **Artificial Characters**: Murphy's artificial characters

## 📈 Metrics and Visualization

### Monitored Metrics
- **Accuracy**: Classification accuracy
- **Loss**: Training loss
- **OIoU Metric**: Overlapping percentage between neurons
- **Validation Accuracy**: Validation set accuracy

### Available Visualizations
```python
# Plot training statistics
pltU.plot_stats(hist.history, hist_IoU_TF=epoch_callback_weigths,
                loss_type=loss_type, datasetName=name_file)

# Visualize decision boundaries
pltU.plot_decision_boundaries(model, X, Y, n_class)

# Compare Normal Loss vs OIoU Loss
pltU.plot_Normal_vs_OIou_Loss(dict_NormalLoss, dict_OIoULoss, dict_path_save)
```

## 🎯 Expected Results

The system typically achieves:
- **Parameter reduction**: 20-50% fewer neurons
- **Accuracy maintenance**: ±2% of original model
- **Generalization improvement**: Reduced overfitting
- **Interpretability**: Neuron overlapping visualization

### Sample Output

```
Dataset Spiral_2_loop_1
Classification rate: 0.9845
OIoU Metric        : 45.23

-----------Output Algorithm 1 Reducing Network size --------------
Original architecture: 402 --> new architecture: 285
Original class rate:   0.9845 --> new class rate   : 0.9867
Original OIoU: 45.23 --> new OIoU: 32.15

-----------Output Algorithm 2 OIoU Loss --------------
Original Classification rate : 0.9867 --> new Classification rate: 0.9912
Architecture: 285
Best IoU_LR --> 0.1
```

## 🔧 Customization

### Adding New Dataset

1. Implement loading function in `DatasourceUtils.py`:
```python
def load_custom_dataset():
    # Load and preprocess data
    # Return: train_dataset, test_dataset, (P, T), (Ptest, Ttest), [input_dim, num_classes, 'DATA']
```

2. Add classification function in `ModelUtils.py`:
```python
def classify_custom_dataset():
    # Load dataset
    # Configure model
    # Execute optimization algorithms
```

### Modifying Architecture

In `BuildModel.py`, customize model construction:
```python
def build_custom_model(neurons, activation, input_shape, output_shape):
    model = Sequential()
    # Add custom layers
    return model
```

## 📝 Technical Notes

### Performance Optimizations
- Use of `@tf.function(jit_compile=True)` for JIT compilation
- Vectorized operations for OIoU calculation
- Batch processing for large datasets

### Memory Considerations
- For large datasets (MNIST, IMDB), consider reducing `batch_size`
- Monitor GPU usage with `nvidia-smi` during training

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## 📚 References

If you use this code in your research, please cite:
```bibtex
@software{oiou_metrics,
  title = {IoU-Based Overlapping Metric and Aggregated Loss Function for Morphological-Linear Neural Networks},
  author = {Gerardo Hernández-Hernández},
  year = {2025},
  url = {https://github.com/GerardoHH/OIoU_Metrics}
}
```

## 📧 Contact

For questions and support, please open an issue in the repository or contact gerardohernandez.hernandez@gmail.com

---

