import numpy as np
# from skimage.transform import resize
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import XGate
from qiskit.circuit.library import QFT
from qiskit.result import marginal_counts
import matplotlib.pyplot as plt

def theta(val): #linear transformation to get theta from pixel value 
    return val * np.pi/2


def im_convert(im): #convert whole image in angles values

    conv = np.zeros(im.shape)

    for i,j in np.ndindex(im.shape):
        conv[i,j] = theta(im[i,j])
    
    return conv
    
def add_xs(qc,bin): #add xs for position

    b_list = []

    for i in bin:
        b_list.append(int(i))
    

    for i in range(len(b_list)):
        if b_list[i] == 0:
            qc.x(-i-1)


def frqi_circuit(im): #for square image 2^n * 2^n

    n = int(np.log2(len(im)))

    thetas = im.flatten()
    
    color = QuantumRegister(1, name='color')
    position =  QuantumRegister(2*n, name='position')

    qc = QuantumCircuit(color, position)

    qc.h(position[0:2*n])
    qc.barrier()

    i = 0
    for theta in thetas:

        bin = format(i, '0'+str(2*n)+'b')

        add_xs(qc, bin)
        qc.mcry(2*theta, position[0:2*n], color[0])
        add_xs(qc, bin)

        qc.barrier()

        i+=1
    
    return qc

def decode_out(qc,shot=2**16):
    
    n = qc.num_qubits

    shots = shot
    backend = Aer.get_backend('qasm_simulator')
    results = execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()

    qubits_of_interest = list(range(1,n))
    marginalised_results = marginal_counts(results, indices=qubits_of_interest)
    marginalised_counts = marginalised_results.get_counts()

    outim = np.zeros((int(2**((n-1)/2)), int(2**((n-1)/2))))

    b = 0
    for i,j in np.ndindex(outim.shape):

        bit = format(b, '0' + str(n-1) + 'b')
        p_tot = marginalised_counts.get(bit,0)
        
        if p_tot == 0:
            pix_val = 0
        else:
            p_i = answer.get(bit+'0',0)

            pix_val = np.arccos(np.sqrt(p_i/p_tot)) * 2/np.pi
        
        outim[i,j] = (pix_val)

        b+=1
    
    return outim

def MSE(im1,im2):

    im = (im1-im2)**2
    s = np.sum(im.flatten())/(len(im1**2))

    return s

def diff_rel(im1,im2):

    
    im = np.abs(im1 - im2)
    s = np.sum(im.flatten())/(len(im1)**2)

    return s * 100











