import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import XGate
from qiskit.circuit.library import QFT
from qiskit.result import marginal_counts
import matplotlib.pyplot as plt


def frqi_circuit(im): #for square image 2^n * 2^n

    n = int(np.log2(len(im)))

    vals = im.flatten()
    vals = vals/np.linalg.norm(vals)
    print(vals)

    qc = QuantumCircuit(2*n)

    qc.prepare_state(vals)

    # qc.measure_all()

    return qc

def decode_out(qc,norm):
    
    n = qc.num_qubits
    shots = 10000

    backend = Aer.get_backend('qasm_simulator')
    results = execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()
    plot_histogram(answer)
    # print(answer)

    outim = np.zeros((int(2**(n/2)), int(2**(n/2))))

    b = 0
    for i,j in np.ndindex(outim.shape):

        bit = format(b, '0' + str(n) + 'b')
        p_i = answer.get(bit,0)/shots

        pix_val = np.sqrt(p_i) * norm
        outim[i,j] = (pix_val)

        b+=1
    
    return outim

def diff_rel(im1,im2):

    
    im = np.abs(im1 - im2)
    s = np.sum(im.flatten())/(len(im1**2))

    return s * 100/255



im = np.array([[0,255,0,255],[255,0,255,0],[0,255,0,255],[255,0,255,0]])
norm = (np.linalg.norm(im.flatten()))
print(norm)
print(im)
plt.imshow(im,cmap='gray', vmin=0, vmax=255)

qc = frqi_circuit(im)
qc.append(QFT(2,insert_barriers=True, name='QFT'),[0,1])
qc.append(QFT(2,insert_barriers=True, name='QFT'),[2,3])
qc.measure_all()
qc.draw('mpl')

# backend = Aer.get_backend('qasm_simulator')
# results = execute(qc, backend=backend, shots=10000).result()
# answer = results.get_counts()

# plot_histogram(answer)

imout = decode_out(qc,norm)
plt.figure()
plt.imshow(imout,cmap='gray', vmin=0, vmax=255)
print(imout)

# print(diff_rel(im,imout))

plt.show()