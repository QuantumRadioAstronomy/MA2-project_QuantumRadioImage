import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, execute, transpile
import qoie as qpie
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import time

def swap_test(state1, state2): #Function to perform swap test between 2 given states
    
    anc = QuantumRegister(1,name='ancilla') #ancilla qubit 
    meas = ClassicalRegister(1) #measurement
    state_1 = QuantumRegister(state1.num_qubits,name='state1') #register for state 1
    state_2 = QuantumRegister(state2.num_qubits,name='state2') #and state 2

    qc = QuantumCircuit(anc,state_1,state_2,meas) #define the quantum circuit

    #Then we just apply SWAP test : H on ancilla, load the states, SWAPS and H on ancilla
    qc.h(anc) 
    qc.append(state1,state_1)
    qc.append(state2,state_2)

    for i in range(1,state1.num_qubits+1):
        qc.cswap(0,i,i+state1.num_qubits)
    
    qc.h(anc)
    qc.measure(anc,meas) #meaasure ancilla

    return qc

class swap_calib():

    def __init__(self, V_ij, V_ijtilda, learn_param = 0.01, nloops = 1000, shift = np.pi/2):

        self.V_ij = V_ij
        self.V_ijtilda = V_ijtilda
        self.learn = learn_param
        self.nloops = nloops
        self.shift = shift
        self.cost = []
        self.steps = []
        self.V_ijtilda_qc = qpie.qpie_circuit(V_ijtilda)
    
    def cost_function(self,params): #cost function : return proba of ancilla being in |1>.

        G_ = np.outer(params,params.T)
        V_ij_ = G_ * self.V_ij #compute with actual parameters

        state1 = qpie.qpie_circuit(V_ij_)
        qc = swap_test(state1,self.V_ijtilda_qc) 

        backend = Aer.get_backend('qasm_simulator')
        results = execute(qc, backend=backend, shots=1024).result()
        answer = results.get_counts() #do simulation

        return answer.get('1',0)/1024 #return proba
    
    def gradient_function(self,params): #return the gradients, computed with shift rule

        gradient = np.zeros(params.shape[0])
        shift = self.shift

        for i in range(params.shape[0]):
            shift_p = params.copy()
            shift_p[i] += shift

            shift_m = params.copy()
            shift_m[i] -= shift

            cost_p = self.cost_function(shift_p)
            cost_m = self.cost_function(shift_m)

            gradient[i] = 1/2 * (cost_p - cost_m)

        return gradient
    
    def grad_desc(self,params): #loop for gradient descent

        for i in range(self.nloops):
            params = params - self.learn * self.gradient_function(params)
            self.cost.append(self.cost_function(params))
            self.steps.append(i)

        return params/np.linalg.norm(params)
    
    def class_opti(self, params): #classical optimizer

        res = minimize(self.cost_function, params, method='COBYLA', tol=1e-9)
        
        return res.x/np.linalg.norm(res.x), res.nfev



