import pickle
import random
from multiprocessing import Pool, freeze_support

import pygsti
import numpy as np
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan2, sqrt, exp
from numpy.linalg import norm
from numpy.random import default_rng
from scipy.optimize import minimize

# from mpl_toolkits.mplot3d import Axes3D

from reset import QuantumSim, prep_projector_matrices_pp
from reset import DensityMatrixOperations as rho_op
#from qiskit.visualization import plot_bloch_multivector, plot_bloch_vector
from math_objects import *
from setonix_job_functions import *

vaccine_parameters = {(4,
  7): {2: [array([-0.91824848, -0.38378502,  0.09761551]),
   0.8080074097190293]},
 (7,
  4): {2: [array([-0.85238576, -0.49620328,  0.16498736]),
   0.7624691070435593]}}

# x2.5 on single-H parameters, unchanged on other paramters (4 SQs)
rotation_axes_list = []
for pair, error_dict in vaccine_parameters.items():
    for qubit, error_info in error_dict.items():
        rotation_axes_list.append(error_info[0])
#print(np.array(rotation_axes_list))

rotation_axes_averaged_list = np.mean([rotation_axes_list[:1], rotation_axes_list[1:]], axis=0)
#print(rotation_axes_averaged_list)

euclid_mean_axis = np.mean(np.array(rotation_axes_list), axis=0)
euclid_mean_axis/= norm(euclid_mean_axis)
print(f'Euclidean mean axis is: {euclid_mean_axis}')

#cons = {'type':'eq', 'fun': con}
#max_iters = {'maxiter':3000}
#optimal_axis_data = minimize(total_distance, euclid_mean_axis, args=np.array(rotation_axes_list),
#                             constraints=cons)

#optimal_axis = optimal_axis_data['x']
#print(f'Spherical mean axis is: {optimal_axis}')
#print(optimal_axis_data)



#rotate_qubits = [2,3,5,8]
#rotate_qubits = [2,3]
rotate_qubits = [2]
euler_angles_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                               for q in rotate_qubits}
euler_angles_reversed_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                        for q in rotate_qubits}

euler_angles_dd_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                      for q in rotate_qubits}
for j in range(len(rotate_qubits)):
    axis = rotation_axes_averaged_list[j]
    Xtalk_rotation_axis = axis/norm(euclid_mean_axis)
    Xtalk_axis_as_thetaphi = asSpherical(Xtalk_rotation_axis)[1:]
    #print(f'rotation axis theta, phi: {Xtalk_axis_as_thetaphi}')

    state_preparation_azimuth_axis = Xtalk_axis_as_thetaphi[1]+pi/2
    state_preparation_azimuth_axis_xyz = asCartesian([1, pi/2, state_preparation_azimuth_axis])
    #print(f'rotation axis xyz for amplifying circuit preparation {state_preparation_azimuth_axis_xyz}')
    state_preparation_angle = Xtalk_axis_as_thetaphi[0]
    #print(f'amplifying circuit preparation angle {state_preparation_angle}')

    euler_angles = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, state_preparation_angle)
    euler_angles_reversed = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, -state_preparation_angle)
    #print(euler_angles)
    #print(euler_angles_reversed)

    #euler_angles_decompositions = [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)]
    for i in range(3):
        angle = euler_angles[i]
        floor_pi10_counts = int(np.floor(abs(angle/(pi/10))))
        round_pi160_counts = round((abs(angle) - pi/10*floor_pi10_counts)/(pi/160))
        if angle > 0:
            euler_angles_decompositions[rotate_qubits[j]][i]['pi10'] = floor_pi10_counts
            euler_angles_decompositions[rotate_qubits[j]][i]['pi160'] = round_pi160_counts
        else:
            euler_angles_decompositions[rotate_qubits[j]][i]['-pi10'] = floor_pi10_counts
            euler_angles_decompositions[rotate_qubits[j]][i]['-pi160'] = round_pi160_counts
    #print(euler_angles_decompositions)


    #euler_angles_reversed_decompositions = [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)]
    for i in range(3):
        angle = euler_angles_reversed[i]
        floor_pi10_counts = int(np.floor(abs(angle/(pi/10))))
        round_pi160_counts = round((abs(angle) - pi/10*floor_pi10_counts)/(pi/160))
        if angle > 0:
            euler_angles_reversed_decompositions[rotate_qubits[j]][i]['pi10'] = floor_pi10_counts
            euler_angles_reversed_decompositions[rotate_qubits[j]][i]['pi160'] = round_pi160_counts
        else:
            euler_angles_reversed_decompositions[rotate_qubits[j]][i]['-pi10'] = floor_pi10_counts
            euler_angles_reversed_decompositions[rotate_qubits[j]][i]['-pi160'] = round_pi160_counts
    #print(euler_angles_reversed_decompositions)
    
    euler_angles_dd = QuantumSim.euler_angles_from_axis_angle(axis, pi)
        
    for i in range(3):
        angle = euler_angles_dd[i]
        floor_pi10_counts = int(np.floor(abs(angle/(pi/10))))
        round_pi160_counts = round((abs(angle) - pi/10*floor_pi10_counts)/(pi/160))
        if angle > 0:
            euler_angles_dd_decompositions[rotate_qubits[j]][i]['pi10'] = floor_pi10_counts
            euler_angles_dd_decompositions[rotate_qubits[j]][i]['pi160'] = round_pi160_counts
        else:
            euler_angles_dd_decompositions[rotate_qubits[j]][i]['-pi10'] = floor_pi10_counts
            euler_angles_dd_decompositions[rotate_qubits[j]][i]['-pi160'] = round_pi160_counts

def GHZ_random_test(Xtalk_count=1, trials=2, sq_num=4):
    shots = 250
    Xtalk_rate = 0.5
    #trials=1
    unfiltered_fidelity_list = []
    filtered_fidelity_list = []
    basis_list = ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']


    ideal_rho_list = []
    random_gates_list = []
    random_gates_names_list = []
    measured_counts_list = [{basis:{bin(i)[2:].zfill(2): {0: 0, 1: 0} 
                                    for i in range(4)} 
                             for basis in basis_list} 
                            for i in range(trials)]


    for x in range(trials):
        sample = QuantumSim(labels=[2,3,4,5,7,8])
        for i in range(1,8):
            sample.random_evolution_2q(4,7)
        random_gates = sample.rand_gates[(4,7)].copy()
        random_gate_names = sample.rand_gate_names[(4,7)].copy()
    
        ideal_state = np.array([1,0,0,0], dtype=complex)
        ideal_state = np.reshape(ideal_state, (4,1))
        for gate in random_gates:
            ideal_state = gate@ideal_state
        ideal_rho = ideal_state@ideal_state.conj().T

        random_gates_list.append(random_gates)
        random_gates_names_list.append(random_gate_names)
        ideal_rho_list.append(ideal_rho)

    with Pool(trials*9) as p:
        print(p)
        if sq_num == 4:
            print('4 spectator qubits used in GHZ detector')
            results = p.starmap(run_crosstalk_GHZ_4SQ_random_gates_trial, 
                                [(shots, basis, euler_angles_decompositions, euler_angles_reversed_decompositions, euler_angles_dd_decompositions, random_gate_names, Xtalk_count, Xtalk_rate) 
                                 for basis in basis_list for random_gate_names in random_gates_names_list])
        elif sq_num == 2:
            print('2 spectator qubits used in GHZ detector')
            results = p.starmap(run_crosstalk_GHZ_2SQ_random_gates_trial, 
                                [(shots, basis, euler_angles_decompositions, euler_angles_reversed_decompositions, euler_angles_dd_decompositions, random_gate_names, Xtalk_count, Xtalk_rate) 
                                 for basis in basis_list for random_gate_names in random_gates_names_list])

        elif sq_num == 1:
            print('1 spectator qubit used in GHZ detector')
            results = p.starmap(run_crosstalk_GHZ_1SQ_random_gates_trial, 
                                [(shots, basis, euler_angles_decompositions, euler_angles_reversed_decompositions, euler_angles_dd_decompositions, random_gate_names, Xtalk_count, Xtalk_rate) 
                                 for basis in basis_list for random_gate_names in random_gates_names_list])
            
        for i in range(len(results)):
            measured_counts_list[i//9][basis_list[i%9]] = results[i]
            print(results[i])


    for x in range(trials):
        unfiltered_normalised_counts, filtered_normalised_counts = rho_op.Xtalk_filter(measured_counts_list[x])
        unfiltered_rho = rho_op.rho_from_dict(unfiltered_normalised_counts)
        filtered_rho = rho_op.rho_from_dict(filtered_normalised_counts)
    
        unfiltered_fidelity = rho_op.fidelity(unfiltered_rho, ideal_rho)
        filtered_fidelity = rho_op.fidelity(filtered_rho, ideal_rho)
    
        unfiltered_fidelity_list.append(unfiltered_fidelity)
        filtered_fidelity_list.append(filtered_fidelity)

    print(unfiltered_fidelity_list)
    print(filtered_fidelity_list)


def old_random_test(Xtalk_count=1, trials=2):
    shots = 250
    Xtalk_rate = 0.5
    #trials = 4
    wait_time = 4
    threshold = 0.0001
    unfiltered_fidelity_list = []
    filtered_fidelity_list = []
    basis_list = ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']

    ideal_rho_list = []
    random_gates_list = []
    random_gates_names_list = []
    measured_counts_list = [{basis:{bin(i)[2:].zfill(2): {0: 0, 1: 0} 
                                    for i in range(4)} 
                             for basis in basis_list} 
                            for i in range(trials)]


    for x in range(trials):
        sample = QuantumSim(labels=[2,3,4,5,7,8])
        for i in range(1,8):
            sample.random_evolution_2q(4,7)
        random_gates = sample.rand_gates[(4,7)].copy()
        random_gate_names = sample.rand_gate_names[(4,7)].copy()
    
        ideal_state = np.array([1,0,0,0], dtype=complex)
        ideal_state = np.reshape(ideal_state, (4,1))
        for gate in random_gates:
            ideal_state = gate@ideal_state
        ideal_rho = ideal_state@ideal_state.conj().T

        random_gates_list.append(random_gates)
        random_gates_names_list.append(random_gate_names)
        ideal_rho_list.append(ideal_rho)

    with Pool(trials*9) as p:
        print(p)
        results = p.starmap(run_crosstalk_old_random_gates_trial, 
                            [(shots, basis, random_gate_names, threshold, wait_time, Xtalk_count, Xtalk_rate) 
                             for basis in basis_list for random_gate_names in random_gates_names_list])
    
        for i in range(len(results)):
            measured_counts_list[i//9][basis_list[i%9]] = results[i]
            print(results[i])


    for x in range(trials):
        unfiltered_normalised_counts, filtered_normalised_counts = rho_op.Xtalk_filter(measured_counts_list[x])
        unfiltered_rho = rho_op.rho_from_dict(unfiltered_normalised_counts)
        filtered_rho = rho_op.rho_from_dict(filtered_normalised_counts)
    
        unfiltered_fidelity = rho_op.fidelity(unfiltered_rho, ideal_rho)
        filtered_fidelity = rho_op.fidelity(filtered_rho, ideal_rho)
    
        unfiltered_fidelity_list.append(unfiltered_fidelity)
        filtered_fidelity_list.append(filtered_fidelity)

    print(unfiltered_fidelity_list)
    print(filtered_fidelity_list)



def GHZ_IDT_test(trials=4, Xtalk_count=1, sq_num=4):
    shots = 250
    Xtalk_rate = 0.5
    trials = 4
    measured_counts_list = []
    with Pool(trials) as p:
        print(p)
        if sq_num == 4:
            print('4 spectator qubits used in GHZ detector')
            for result in p.starmap(run_crosstalk_GHZ_IDT_trial_4SQ,
                                    [(shots, euler_angles_decompositions, euler_angles_reversed_decompositions, euler_angles_dd_decompositions, Xtalk_count, Xtalk_rate) for i in range(trials)]):
                measured_counts_list.append(result)
        
        
        elif sq_num == 2:
            print('2 spectator qubits used in GHZ detector')
            for result in p.starmap(run_crosstalk_GHZ_IDT_trial_2SQ,
                                    [(shots, euler_angles_decompositions, euler_angles_reversed_decompositions, euler_angles_dd_decompositions, Xtalk_count, Xtalk_rate) for i in range(trials)]):
                measured_counts_list.append(result)
                
        else:
            print('1 spectator qubit used in GHZ detector')
            for result in p.starmap(run_crosstalk_GHZ_IDT_trial_1SQ,
                                    [(shots, euler_angles_decompositions, euler_angles_reversed_decompositions, euler_angles_dd_decompositions, Xtalk_count, Xtalk_rate) for i in range(trials)]):
                measured_counts_list.append(result)

    crosstalk_included_list = []
    crosstalk_removed_list = []
    for i in range(trials):
        crosstalk_included = []
        crosstalk_removed = []
        for data_outcome, count in measured_counts_list[i].items():
            crosstalk_included.append(count[0]+count[1])
            crosstalk_removed.append(count[0])
    
        crosstalk_included = np.asarray(crosstalk_included)
        crosstalk_included = crosstalk_included/np.sum(crosstalk_included)
        crosstalk_removed = np.asarray(crosstalk_removed)
        crosstalk_removed = crosstalk_removed/np.sum(crosstalk_removed)

        crosstalk_included_list.append(crosstalk_included[0])
        crosstalk_removed_list.append(crosstalk_removed[0])

    #crosstalk_included_mean = np.mean(crosstalk_included_list, axis=0)
    #crosstalk_included_std = np.std(crosstalk_included_list, axis=0)/np.sqrt(len(crosstalk_included_list))
    #crosstalk_removed_mean = np.mean(crosstalk_removed_list, axis=0)
    #crosstalk_removed_std = np.std(crosstalk_removed_list, axis=0)/np.sqrt(len(crosstalk_removed_list))
    #print(1-crosstalk_included_mean)
    #print(crosstalk_included_std)
    #print(1-crosstalk_removed_mean)
    #print(crosstalk_removed_std)

    print(1-np.array(crosstalk_included_list))
    print(1-np.array(crosstalk_removed_list))
    

def old_IDT_test(trials=4, Xtalk_count=1):

    shots = 250
    Xtalk_rate = 0.5
    #trials = 4
    wait_time = 4
    threshold = 0.0001
    
    measured_counts_list = []
    with Pool(trials) as p:
        print(p)
        for result in p.starmap(run_crosstalk_old_IDT_trial,
                                [(shots,threshold, wait_time, Xtalk_count, Xtalk_rate) for i in range(trials)]):
            measured_counts_list.append(result)


    crosstalk_included_list = []
    crosstalk_removed_list = []
    for i in range(trials):
        crosstalk_included = []
        crosstalk_removed = []
        for data_outcome, count in measured_counts_list[i].items():
            crosstalk_included.append(count[0]+count[1])
            crosstalk_removed.append(count[0])
    
        crosstalk_included = np.asarray(crosstalk_included)
        crosstalk_included = crosstalk_included/np.sum(crosstalk_included)
        crosstalk_removed = np.asarray(crosstalk_removed)
        crosstalk_removed = crosstalk_removed/np.sum(crosstalk_removed)

        crosstalk_included_list.append(crosstalk_included[0])
        crosstalk_removed_list.append(crosstalk_removed[0])

    #crosstalk_included_mean = np.mean(crosstalk_included_list, axis=0)
    #crosstalk_included_std = np.std(crosstalk_included_list, axis=0)/np.sqrt(len(crosstalk_included_list))
    #crosstalk_removed_mean = np.mean(crosstalk_removed_list, axis=0)
    #crosstalk_removed_std = np.std(crosstalk_removed_list, axis=0)/np.sqrt(len(crosstalk_removed_list))
    #print(1-crosstalk_included_mean)
    #print(crosstalk_included_std)
    #print(1-crosstalk_removed_mean)
    #print(crosstalk_removed_std)

    print(1-np.array(crosstalk_included_list))
    print(1-np.array(crosstalk_removed_list))

#test()


if __name__ == '__main__':
    freeze_support()
    #for i in range(8):
    #    GHZ_random_test(Xtalk_count=4, trials=1, sq_num=1)
    #for i in range(8):
    #    GHZ_random_test(Xtalk_count=6, trials=1, sq_num=2)

    #for i in range(8):
    #    old_random_test(Xtalk_count=0, trials=1)
    for i in range(8):
        old_random_test(Xtalk_count=4, trials=1)