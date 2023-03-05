import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import XGate
from qiskit.circuit.library import QFT
from qiskit.result import marginal_counts
import matplotlib.pyplot as plt

def theta(val): #linear transformation to get theta from pixel value 
    return val/255 * np.pi/2


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

    # qc.measure_all()
    
    return qc

def decode_out(qc):
    
    n = qc.num_qubits
    # print(n)

    backend = Aer.get_backend('qasm_simulator')
    results = execute(qc, backend=backend, shots=2**16).result()
    answer = results.get_counts()
    # print(answer)
    qubits_of_interest = list(range(1,n))
    marginalised_results = marginal_counts(results, indices=qubits_of_interest)
    marginalised_counts = marginalised_results.get_counts()
    # print(marginalised_counts)

    outim = np.zeros((int(2**((n-1)/2)), int(2**((n-1)/2))))

    b = 0
    for i,j in np.ndindex(outim.shape):

        bit = format(b, '0' + str(n-1) + 'b')
        p_tot = marginalised_counts.get(bit)
        p_i = answer.get(bit+'0')

        if p_i == None:
            p_i = 0

        pix_val = np.arccos(np.sqrt(p_i/p_tot)) * 255 * 2/np.pi
        outim[i,j] = round(pix_val)

        b+=1
    
    return outim



im = np.random.randint(0,256,(4,4))
print(im)
plt.imshow(im,cmap='gray', vmin=0, vmax=255)
plt.show()

conv = im_convert(im)
qc = frqi_circuit(conv)
qc.append(QFT(2,insert_barriers=True, name='QFT'),[1,2])
qc.measure_all()

# outim = decode_out(qc)
# print(outim)
# plt.imshow(outim,cmap='gray', vmin=0, vmax=255)
# plt.show()

outim = decode_out(qc)
print(outim)
plt.imshow(outim,cmap='gray', vmin=0, vmax=255)
plt.show()

conv_ = im_convert(outim)
qc2 = frqi_circuit(conv_)
qc2.append(QFT(2,insert_barriers=True, name='QFT').inverse(),[1,2])
qc2.measure_all()

outim2 = decode_out(qc2)
print(outim2)
plt.imshow(outim2,cmap='gray', vmin=0, vmax=255)
plt.show()


# qc.draw('mpl')
# plt.show()









