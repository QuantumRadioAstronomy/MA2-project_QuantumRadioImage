import numpy as np
from qiskit import QuantumCircuit, Aer, execute


def qpie_circuit(im): #for square image 2^n * 2^n

    n = int(np.log2(len(im)))

    vals = im.flatten()
    vals = vals/np.linalg.norm(vals)

    qc = QuantumCircuit(2*n)
    qc.initialize(vals)

    return qc

def decode_out(qc, norm, shot=10000, fourier=False):
    
    n = qc.num_qubits
    shots = shot

    backend = Aer.get_backend('qasm_simulator')
    results = execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()

    outim = np.zeros((int(2**(n/2)), int(2**(n/2))))

    b = 0
    for i,j in np.ndindex(outim.shape):

        bit = format(b, '0' + str(n) + 'b')
        p_i = answer.get(bit,0)/shots

        if fourier:
            pix_val = np.sqrt(p_i) * norm * 2**(n/2)
        else:
            pix_val = np.sqrt(p_i) * norm

        outim[i,j] = (pix_val)

        b+=1
    
    return outim

def diff_rel(im1,im2):

    
    im = np.abs(im1 - im2)
    s = np.sum(im.flatten())/(len(im1)**2)

    return s * 100

def MSE(im1,im2):

    im = (im1-im2)**2
    s = np.sum(im.flatten())/(len(im1)**2)

    return s
