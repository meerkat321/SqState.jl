<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## Source code status

![CI](https://github.com/foldfelis/SqState.jl/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/foldfelis/SqState.jl/branch/master/graph/badge.svg?token=LIVF96N05K)](https://codecov.io/gh/foldfelis/SqState.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

## Render Wigner function

The Wigner function is calculate by Moyal function in Fock basis

$$
W_{mn}(x, p) = \frac{1}{2\pi} \int_{-\infty}^{\infty} dy e^{-ipy/h} \psi_m^*(x+\frac{y}{2}) \psi_n(x-\frac{y}{2})
$$

Owing to the Moyal function is a generalization of the Wigner function. We can therefore implies that

$$
W(x, p) = \sum_{m, n} \rho_{m, n} W_{m, n}(x, p)
$$

The squeezed vacuum state shown as below

![Wigner function surface](assets/index/wigner_surface_banner.png)

## Example

* [Plot Wigner function](notebook/plot_wigner.jl.html)
* [Plot Fock state Wigner function](notebook/fock_state.jl.html)
