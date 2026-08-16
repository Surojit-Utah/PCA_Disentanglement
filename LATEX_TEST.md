# GitHub LaTeX Test

## Test 1: Simple subscripts (no bold)
The factorized Gaussian posterior $q_\phi(z | x) = \mathcal{N}(\mu_x, \sigma_x^2 I)$ encourages axis-aligned representations.

## Test 2: Bold vectors, simple subscripts
The factorized Gaussian posterior $q_\phi(\mathbf{z} | \mathbf{x}) = \mathcal{N}(\mu_x, \sigma_x^2 I)$ encourages axis-aligned representations.

## Test 3: Using \mid instead of |
The factorized Gaussian posterior $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\mu_x, \sigma_x^2 I)$ encourages axis-aligned representations.

## Test 4: Bold subscripts with underscore only
The factorized Gaussian posterior $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_x, \boldsymbol{\sigma}_x^2\mathbf{I})$ encourages axis-aligned representations.

## Test 5: Current version (with \mathbf in subscript)
The factorized Gaussian posterior $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\mathbf{x}, \boldsymbol{\sigma}_\mathbf{x}^2\mathbf{I})$ encourages axis-aligned representations.

## Test 6: Using text for subscripts
The factorized Gaussian posterior $q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_{\text{x}}, \boldsymbol{\sigma}_{\text{x}}^2\mathbf{I})$ encourages axis-aligned representations.

## Display Math Test
$$
\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I}) \quad \Rightarrow \quad \text{rot}(\mathbf{z}) \sim \mathcal{N}(\mathbf{0}, \mathbf{I})
$$

## Aggregate posterior
Match the aggregate posterior $q_\phi(\mathbf{z})$ to isotropic prior $\mathcal{N}(\mathbf{0}, \mathbf{I})$
