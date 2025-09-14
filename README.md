# IoU-Based Overlapping Metric and Aggregated Loss Function for Morphological-Linear Neural Networks

## Description
This project implements a novel neural network optimization framework that uses Dendral Neurons and the Overlapping Index over Union (OIoU) metric to automatically reduce network complexity while maintaining classification performance. The system employs two optimization algorithms that progressively reduce the number of neurons based on overlapping measurements between hypercube-based neuron representations, achieving 20-50% parameter reduction with minimal accuracy loss.

## Dataset Information
All datasets were processed to have a zero mean distribution and a standard deviation of 1.

### Synthetic Datasets
- **Dataset A**: Binary classification problem with 2D features, 1000 training samples, 1000 test samples
- **Dataset B**: 3-class classification problem with 2D features
- **XOR Dataset**: Non-linearly separable XOR problem with Gaussian noise, 2D features
- **Spiral Dataset**: Configurable spiral patterns with:
  - 2 or 3 classes
  - 1, 2, or 5 loops/spirals
  - Balanced class distribution

### Real-World Datasets
- **MNIST**: 60,000 training images, 10,000 test images, 784 features (28x28 pixels), 10 classes      DOI: 10.24432/C53K8Q
- **Letter Recognition**: 20,000 samples, 16 numerical features, 26 classes (A-Z)                     DOI: 10.24432/C5ZP40
- **IMDB Movie Reviews**: 50,000 reviews for sentiment analysis, Word2Vec embeddings (100 dimensions) URL: http://ai.stanford.edu/~amaas/data/sentiment/
- **Artificial Characters**: Murphy's artificial character dataset with 36 features, 10 character classes DI: 10.24432/C5303Z

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

## Usage Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/GerardoHH/OIoU_Metrics.git
cd OIoU_Metrics

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
- Minimum 16GB RAM for text processing tasks
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
  url = {https://github.com/GerardoHH/OIoU_Metrics},
  note = {Software available at: https://github.com/GerardoHH/OIoU_Metrics.git}
}
```


## License & Contribution Guidelines

### License
This project is licensed under the MIT License. See `LICENSE` file for details.

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


## 📧 Contact

For questions and support, please open an issue in the repository or contact gerardohernandez.hernandez@gmail.com

---

