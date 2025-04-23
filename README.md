# RNN-InceptionTime-MoE Model for Spatiotemporal Diagnosis of Parkinson’s disease on VGRF Data

## Hyperparameters

### General  
These settings are shared across all Expert, Gate, and MoE models:

**RNN**  
- Input size: 500 (window size)  
- Hidden state size: 128  
- Number of layers: 1  
- Dropout probability: 0.0  
- Bidirectional: Yes  

**InceptionTime**  
- Input size: 16 (number of features)  
- Output size per filter/convolution: 32  
- Kernel sizes for each filter/convolution: [39, 19, 9]  
- Depth: 6  
- Residual connections: Yes  

**Fully-Connected (FC)**  
- Dropout probability: 0.0  

### Expert and MoE Models  
- Output size: 4 (labels: Healthy, Severity-2, Severity-2.5, Severity-3)

### Gate Model
- Output size: 3 (labels: Ga, Ju, Si)