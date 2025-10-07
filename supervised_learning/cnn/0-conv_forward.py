#!/usr/bin/env python3
"""
Performs forward propagation over a convolutional layer.

This function applies a convolution operation between the input tensor `A_prev`
and a set of learned filters `W`, adds the corresponding biases `b`, and 
applies an activation function to the result. It supports both "same" and 
"valid" padding modes and configurable strides.

Typical use case: this is the forward pass of a convolutional layer in a 
Convolutional Neural Network (CNN).

Example:
    >>> import numpy as np
    >>> def relu(x): return np.maximum(0, x)
    >>> A_prev = np.random.randn(2, 5, 5, 3)
    >>> W = np.random.randn(3, 3, 3, 8)
    >>> b = np.zeros((1, 1, 1, 8))
    >>> A = conv_forward(A_prev, W, b, relu, padding="same", stride=(1, 1))
    >>> A.shape
    (2, 5, 5, 8)
"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation through a convolutional layer.

    Args:
        A_prev (numpy.ndarray): Activations from the previous layer or input data
            of shape (m, h_prev, w_prev, c_prev)
            - m: number of examples
            - h_prev, w_prev: height and width of the input
            - c_prev: number of channels in the input
        W (numpy.ndarray): Weights (filters) tensor of shape (kh, kw, c_prev, c_new)
            - kh, kw: height and width of each kernel
            - c_prev: number of channels in input
            - c_new: number of filters (output channels)
        b (numpy.ndarray): Bias tensor of shape (1, 1, 1, c_new)
        activation (callable): Activation function to apply (e.g., relu, sigmoid, tanh)
        padding (str): Either "same" or "valid"
            - "same": output has the same spatial dimensions as the input
            - "valid": no padding, output shrinks
        stride (tuple): Tuple of (sh, sw) specifying stride height and width

    Returns:
        numpy.ndarray: The output activations of shape (m, h_new, w_new, c_new)
            - h_new = floor((h_prev + 2*ph - kh) / sh) + 1
            - w_new = floor((w_prev + 2*pw - kw) / sw) + 1

    Raises:
        ValueError: If padding is not "same" or "valid".

    Notes:
        - This implementation uses explicit for-loops for educational clarity.
        - No vectorized convolution is performed here (it’s pedagogical, not optimized).
        - The padding is computed to preserve the spatial dimensions when `padding="same"`.

    """
    m, h, w, c = A_prev.shape
    kh, kw, _, kc = W.shape
    sh, sw = stride

    # Step 1: Compute padding
    if padding == "valid":
        ph = 0
        pw = 0
    elif padding == "same":
        ph = (kh - 1) // 2
        pw = (kw - 1) // 2
    else:
        raise ValueError("padding must be 'same' or 'valid'")

    # Step 2: Pad the input volume
    A_prev_padded = np.pad(
        A_prev,
        pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
        mode='constant',
        constant_values=0
    )

    # Step 3: Compute output dimensions
    _, padded_h, padded_w, _ = A_prev_padded.shape
    output_h = (padded_h - kh) // sh + 1
    output_w = (padded_w - kw) // sw + 1

    # Step 4: Initialize output tensor
    A = np.zeros((m, output_h, output_w, kc))

    # Step 5: Perform the convolution
    for i in range(output_h):
        for j in range(output_w):
            for k in range(kc):
                kernel = W[:, :, :, k]
                region = A_prev_padded[:,
                    i * sh : i * sh + kh,
                    j * sw : j * sw + kw,
                    :]
                conv_value =np.sum(region * kernel, axis=(1,2,3)) + b[0, 0, 0, k]
                A[:, i, j, k] = activation(conv_value)

    return A
