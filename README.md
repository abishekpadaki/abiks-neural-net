# abiks-neural-net
My experiments with building and designing neural nets

## Project Structure

```
abiks-neural-net/
├── attention_in_transformers/          # Transformer attention mechanisms
│   ├── self_attn.ipynb                # Self-attention implementation
│   ├── masked_self_attn.ipynb         # Masked self-attention for language models  
│   └── multi_head_attn.ipynb          # Multi-head attention mechanism
├── covolution_nets/                    # Convolutional neural networks
│   └── how_even_does_alexnet_work.ipynb # AlexNet architecture breakdown
├── mlp_architectures/                  # Multi-layer perceptron architectures
│   └── autoencoder.ipynb              # Autoencoder implementation
├── numpy_is_all_you_need/             # Pure NumPy implementations
│   ├── mnsit/
│   │   └── abiks-mnsit-from-scratch.ipynb # MNIST from scratch with NumPy
│   └── simple_forward.py              # Basic neural network forward pass
└── torchup/                           # PyTorch fundamentals
    └── basic/
        ├── linear_reg.ipynb           # Linear regression with PyTorch
        └── model_def_template.md     # Template for model definitions
```

## What's Inside

- **Attention Mechanisms**: Self-attention, masked attention, and multi-head attention implementations
- **CNNs**: Deep dive into AlexNet architecture and convolutional layers
- **Autoencoders**: Encoder-decoder architectures for dimensionality reduction
- **From Scratch**: Pure NumPy neural networks to understand the fundamentals
- **PyTorch Basics**: Getting started with PyTorch for deep learning
