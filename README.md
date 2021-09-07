# SqState

Squeezed state solver for quantum optics.

[![CI](https://github.com/foldfelis-QO/SqState.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/foldfelis-QO/SqState.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/foldfelis-QO/SqState.jl/branch/master/graph/badge.svg?token=5EFID3REPE)](https://codecov.io/gh/foldfelis-QO/SqState.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)

## About

In this real time squeezed state tomography system, we introduced a machine learning model that illustrates fast and precise squeezed state tomography for continuous variables, through the experimentally measured data generated from the balanced homodyne detectors.

## Model

The data of quadrature sequence obtained by quantum homodyne tomography are fed to the model with its architecture constructed by Fourier operators. Then, after flattening, the predicted arguments represents the squeezed state in concept.

```julia
Chain(
  Conv((1,), 1 => 64),                  # 128 parameters
  FourierOperator(
    Conv((1,), 64 => 64),               # 4_160 parameters
    SpectralConv(64 => 64, (24,), σ=identity, permuted=true),  # 98_304 parameters
    NNlib.gelu,
  ),
  FourierOperator(
    Conv((1,), 64 => 64),               # 4_160 parameters
    SpectralConv(64 => 64, (24,), σ=identity, permuted=true),  # 98_304 parameters
    NNlib.gelu,
  ),
  FourierOperator(
    Conv((1,), 64 => 64),               # 4_160 parameters
    SpectralConv(64 => 64, (24,), σ=identity, permuted=true),  # 98_304 parameters
    NNlib.gelu,
  ),
  FourierOperator(
    Conv((1,), 64 => 64),               # 4_160 parameters
    SpectralConv(64 => 64, (24,), σ=identity, permuted=true),  # 98_304 parameters
    identity,
  ),
  Conv((2,), 64 => 32, gelu, stride=2),  # 4_128 parameters
  Conv((2,), 32 => 16, gelu, stride=2),  # 1_040 parameters
  Conv((4,), 16 => 8, gelu, stride=4),  # 520 parameters
  Conv((4,), 8 => 4, gelu, stride=4),   # 132 parameters
  Flux.flatten,
  Dense(256, 32, gelu),                 # 8_224 parameters
  Dense(32, 6, relu),                   # 198 parameters
)                   # Total: 26 arrays, 424_226 parameters, 3.122 MiB.
```
