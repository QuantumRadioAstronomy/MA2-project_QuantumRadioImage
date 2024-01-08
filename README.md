# Quantum image representation

The repo contains code implementing QPIE and FRQI image encodings and a quantum algorithm for Self Calibration. It uses IBM's Qiskit framework for creating/simulating the circuits.
It is associated with paper Quantum Radio Astronomy: Data Encodings and Quantum Image Processing.

## QPIE.ipynb -- QPIE Encoding and Quantum Fourier Transform

Python notebook showing how QPIE works with different images. Shows how quantum fourier transform is used to obtain the fourier transform of images.

 ## FRQI.py -- FRQI encoding

Implementation of FRQI encoding. Contain functions to convert the image with angle values, construct the quantum circuit and decode the image from a quantum circuit.

## QPIE.py -- QPIE encoding

Implementation of QPIE encoding, contains functions to construct the quantum circuit and to decode the image from the circuit.

## var_swap.py -- Quantum Self Calibration

Code for the self calibration quantum algorithm which uses the swap test algo. Contains a function to perform a swap test between 2 quantum states.
A class to perform the algorithm described in the paper : define the cost function from the swap test and optimize it using gradient descent or COBYLA optimizer from scipy.

## main_var.py

Example on how to use the self calibration algo code. Used to obtain Fig.8 in the paper.

