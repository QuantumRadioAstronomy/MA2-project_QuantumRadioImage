import numpy as np
from skimage.transform import resize
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
        p_tot = marginalised_counts.get(bit,0)
        
        if p_tot == 0:
            pix_val = 0
        else:
            p_i = answer.get(bit+'0',0)

            pix_val = np.arccos(np.sqrt(p_i/p_tot)) * 2/np.pi * 255
        
        outim[i,j] = (pix_val)

        b+=1
    
    return outim

def diff_rel(im1,im2): #for relative difference between in and out

    
    im = np.abs(im1 - im2)
    s = np.sum(im.flatten())/(len(im1**2))

    return s * 100/255

im = np.random.randint(0,256,(16,16)) #generate random image
print(im)
plt.imshow(im,cmap='gray', vmin=0, vmax=255)

conv = im_convert(im) #convert it in angles values
qc = frqi_circuit(conv) #create the circuit
# qc.append(QFT(2,insert_barriers=True, name='QFT'),[1,2])
# qc.append(QFT(2,insert_barriers=True, name='QFT'),[3,4])
qc.measure_all() #measure

imout = decode_out(qc) #decode from measuring circuit
plt.figure()
plt.imshow(imout,cmap='gray', vmin=0, vmax=255)
print(imout)

plt.show()










