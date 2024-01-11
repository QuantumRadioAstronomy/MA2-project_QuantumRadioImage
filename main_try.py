import FRQI as frqi
import QPIE as qpie
import var_swap as cal
import numpy as np
from qiskit.circuit.library import QFT
from scipy.optimize import least_squares as lsq
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['font.size'] = 12

##To use calibration, first empty lists for observations

inits = []
opti_class = []
opti_gd = []
good_g = []

diff_class = []
diff_gd = []
diff_full = []

cost_init = []
cost_gd = []
cost_clas = []
cost_g = []
cost_fullclass = []

it_clas=[]

his_fullclas  = []
his_clas = []
his_q = []

for i in range(50): #number of experience you want

    V_ij = np.random.rand(2,2) #True observation
    g = np.random.rand(2) #True g we want to find
    g = g/np.linalg.norm(g)
    G = np.outer(g,g.T)
    V_ijtilda = G * V_ij #Observed visibilities

    def loss(x): #loss for least squares fct from scipy
        G = np.outer(x,x.T)
        V_reco = G*V_ij
        return (V_reco - V_ijtilda).flatten()

    parameters = np.random.rand(2) #random initial gains

    inits.append(parameters)
    good_g.append(g)

    var = cal.swap_calib(V_ij,V_ijtilda) 

    cost_init.append(var.cost_function(parameters)) #compute cost for init params
    cost_g.append(var.cost_function(g)) #and for good g

    clas,nit = var.class_opti(parameters) #optimize with classical
    gd = var.grad_desc(parameters) #and with QGD
    clas_full = lsq(loss, parameters) #and classical with scipy lsq

    # We add to observations the results

    opti_class.append(clas)
    opti_gd.append(gd)
    it_clas.append(nit)

    cost_clas.append(var.cost_function(clas))
    cost_gd.append(var.cost_function(gd))


    diff_class.append(np.abs(clas-g))
    diff_gd.append(np.abs(gd-g))
    diff_full.append(np.abs(clas_full.x - g))

    his_clas.append(np.abs(clas[0]-g[0]))
    his_clas.append(np.abs(clas[1]-g[1]))
    his_q.append(np.abs(gd[0]-g[0]))
    his_q.append(np.abs(gd[1]-g[1]))
    his_fullclas.append(np.abs(clas_full.x[0]-g[0]))
    his_fullclas.append(np.abs(clas_full.x[1]-g[1]))

    # Uncomment if want QGD minimization over steps

    # plt.figure() #plot cost/step for QGD
    # plt.xlabel('Step',fontsize=15)
    # plt.ylabel('Cost Function', fontsize=15)
    # plt.plot(var.steps,var.cost)


#Prints observations
# print(inits)
# print(opti_class)
# print(opti_gd)
# print(good_g)
# print(diff_class)
# print(diff_gd)
# print(it_clas)
# print(cost_init)
# print(cost_clas)
# print(cost_gd)
# print(cost_g)

# print(his_clas)
# print(his_q)

counts, bins = np.histogram(his_q) #histogram with qgd and hybrid
plt.figure()

plt.xlabel('Absolute Error')
plt.ylabel('Counts')

plt.hist(his_clas, bins = bins, color = 'blue', alpha = 0.5, histtype = 'stepfilled')
plt.hist(his_clas, bins = bins, edgecolor = 'blue', alpha = 1.0, histtype = 'step', label='Hybrid')

plt.hist(his_q, bins = bins, color = 'red', alpha = 0.3, histtype = 'stepfilled')
plt.hist(his_q, bins = bins, edgecolor = 'red', alpha = 1.0, histtype = 'step', label = 'QGD')

plt.legend()

plt.figure() #histogram for full classical

counts, bins = np.histogram(his_fullclas)

plt.xlabel('Absolute Error')
plt.ylabel('Counts')

plt.hist(his_fullclas, bins = bins, color = 'orchid', alpha = 0.3, histtype = 'stepfilled')
plt.hist(his_fullclas, bins = bins, edgecolor = 'orchid', alpha = 1.0, histtype = 'step', label = 'Classical LSQ')

plt.legend()

plt.show()
