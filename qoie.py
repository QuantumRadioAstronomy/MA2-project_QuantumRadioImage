import numpy as np
from skimage.transform import resize
from skimage import io
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import MCMT
from qiskit.circuit.library import XGate
from qiskit.circuit.library import QFT
from qiskit.result import marginal_counts
import matplotlib.pyplot as plt


def qpie_circuit(im): #for square image 2^n * 2^n

    n = int(np.log2(len(im)))

    vals = im.flatten()
    vals = vals/np.linalg.norm(vals)

    print(np.sum(vals**2))

    qc = QuantumCircuit(2*n)

    qc.initialize(vals)

    # qc.measure_all()

    return qc

def decode_out(qc,norm):
    
    n = qc.num_qubits
    shots = 2**20

    backend = Aer.get_backend('qasm_simulator')
    results = execute(qc, backend=backend, shots=shots).result()
    answer = results.get_counts()
    # print(results)
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

    return s * 100

#### Many different images for try in comments, sorry kinda messy


# im = io.imread('lena.png', as_gray=True)
# im = resize(img,(256,256))
# im = plt.imread('lena.png')[:,:,1] * 255
# im = np.random.rand(256,256)
# im = np.array([[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[255,255,255,255,255,255,255,255],[255,255,255,255,255,255,255,255],
            #    [0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[255,255,255,255,255,255,255,255],[255,255,255,255,255,255,255,255]])
# im = np.array([[0,0,0,0],[1,1,1,1],[0,0,0,0],[1,1,1,1]])
im = np.zeros((32,32))
im[5,6] = 1
im[6,6] = 1
im[5,7] = 1
im[6,7] = 1
# im[24,25] = 1
# im[25,25] = 1
# im[24,26] = 1
# im[25,26] = 1

# im[4,4]=1
# im[4,5]=1
# im[5,4]=1
# im[5,5]=1
# im[6,4]=1
# im[6,5]=1
# im[7,4]=1
# im[7,5]=1

# im[5,3]=1
# im[6,3]=1
# im[5,5]=1
# im[6,6]=1
# im[5,6]=1

# im[10:14,10:18] =1
# im[9,11:17]=1
# im[8,12:16]=1
# im[14,11:17]=1
# im[15,12:16]=1

norm = (np.linalg.norm(im.flatten())) #compute norm of image 
print(norm)
print(im)
plt.imshow(im,cmap='gray')

qc = qpie_circuit(im) #get circuit 
## add QFT, care here if image size changed, need to change also here. First apply on y coords then on x coords
qc.append(QFT(5,insert_barriers=True, name='QFT'),[0,1,2,3,4])
qc.append(QFT(5,insert_barriers=True, name='QFT'),[5,6,7,8,9])
qc.measure_all()
# qc.draw('mpl') #if want to see circuit

imout = decode_out(qc,norm)
plt.figure()
plt.imshow(imout,cmap='gray')
print(imout)

plt.show()