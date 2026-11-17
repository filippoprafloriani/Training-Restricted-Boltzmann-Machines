# Training Restricted Boltzmann Machines

Group project for the course *Laboratory of Computational Physics* (University of Padova, January 2024), by Guillermo Benito, Claudia Lorenzetti, Leon Mengoni and Filippo Pra Floriani.

A Restricted Boltzmann Machine (RBM) is implemented **from scratch in NumPy** and trained to learn the distribution of noisy structured data, in order to denoise it and generate new clean samples. The project compares training strategies and studies how to assess the quality of a generative model.

## Model

An RBM is an energy-based generative model with a layer of visible units $v$ (the data) and a layer of hidden units $h$, coupled only across layers.

$$E(v, h) = -\sum_{ij} v_i w_{ij} h_j - \sum_i a_i v_i - \sum_j b_j h_j, \qquad p(v, h) = \frac{e^{-\beta E(v,h)}}{Z}$$

It is the same structure as an Ising model with two coupled layers of spins. Training maximises the log-likelihood of the data, with the model expectation estimated by **contrastive divergence** (CD-n), a truncated Gibbs sampling chain.

## Data

Two synthetic datasets with a known underlying pattern.

- **Block patterns.** 30-bit strings of alternating blocks (`001111` / `111100`), where each bit is flipped with 10% probability. Used to test denoising.
- **Amino acid chains.** Chains of 5 one-hot encoded amino acids (20 bits) alternating polar and non-polar types, where 10% of the blocks have the wrong type. Used to test generation. For this dataset the one-hot structure is enforced in the reconstruction step by sampling each 4-bit block from its Boltzmann (softmax) weights.

## What we studied

- Grid search over optimiser (SGD, RMSprop, Adam), number of hidden units (2, 4, 6, 10) and CD steps (1, 2, 5, 8, 10)
- The centering trick
- Quality indicators, i.e. the adversarial accuracy indicator, log-likelihood and energy of generated vs original data, and a polar / non-polar alternation check

## Results

- At low temperature ($\beta = 40$) the RBM denoises the block patterns perfectly.
- The best model is plain SGD with 2 hidden units and CD-1, with an adversarial accuracy indicator of $4.1 \cdot 10^{-7}$, while larger models tend to overfit this simple dataset.
- Log-likelihood and energy of generated data converge to those of the original data during training.
- The centering trick gives no improvement on this problem.
- The learned weights reflect the block structure of the data.

<!-- Suggested: add the denoising figure and the log-likelihood / energy curves from the report, saved in a figures/ folder -->

## Repository structure

| File | Content |
|---|---|
| `RBM.ipynb` | Implementation, training, grid search and analysis. Settings (dataset, optimiser, hidden units, CD steps, one-hot, temperature) are at the top of the notebook |
| `Assignment_LCP.pdf` | Report with theory, methods and results |

## How to run

```bash
pip install numpy pandas matplotlib seaborn
jupyter notebook RBM.ipynb
```

The notebook reads the datasets from `DATA/` and `DATA_b/` (files `x_RBM_q0.1.dat`).
