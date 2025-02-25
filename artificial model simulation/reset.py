import os
import warnings
import itertools
import functools as ft
import pygsti
import pickle
import numpy as np
from pygsti.circuits import Circuit
from numpy import pi, sin, cos
from numpy.linalg import norm
from scipy.spatial.transform import Rotation
from scipy.linalg import sqrtm
from scipy.linalg import expm
import random
from multiprocessing import Pool
#from multiprocessing.pool import ThreadPool
from math_objects import *

def convert_matrix(x):
    '''
    compute the projected density matrix,
    e is any pauli basis component of the density matrix,
    m is the projector e.g. I*|0><0|*I*I*I
    expect: (I*|0><0|*I*I*I)(X*X*X*Y*Z)(I*|0><0|*I*I*I) = IXI*|0><0|X|0><0|*IXI*IYI*IZI = X*|0><0|X|0><0|*X*Y*Z = <0|X|0> (X*|0><0|*X*Y*Z) (* means kronecker product)
    after stdmx_to_ppvec gives
    [M^\sigma_1_sigma_1,..., M^\sigma_1_sigma_n] (supscript means the basis it projected from, undscrip means its component in this rep)
    '''
    (m, e) = x
    return pygsti.tools.stdmx_to_ppvec(m @ e @ m)

def convert_matrix_asymmetry(x):
    (m1, m2, e) = x
    return pygsti.tools.stdmx_to_ppvec(m1 @ e @ m2)

def convert_matrix_left(x):
    (m, e) = x
    return pygsti.tools.stdmx_to_ppvec(m @ e)

def convert_matrix_right(x):
    (m, e) = x
    return pygsti.tools.stdmx_to_ppvec(e @ m)

def convert_matrix_hermitian(x):
    (m, e) = x
    return pygsti.tools.stdmx_to_ppvec(m @ e @ m.T)

def convert_matrix_hermitian_true(x):
    (m, e) = x
    return pygsti.tools.stdmx_to_ppvec(m @ e @ m.conj().T)

def prep_projector_matrices_pp(num_qubits):
    dm_0 = np.array([[1, 0], [0, 0]], complex)
    dm_1 = np.array([[0, 0], [0, 1]], complex)
    measurement_matrices = []
    #create measurement matrices for each qubit
    for i in range(num_qubits):
        #initialize measurement operator as 1 for measuring qubit i
        full_dm_0 = np.array([1])
        full_dm_1 = np.array([1])

        # Generate POVM
        for j in range(num_qubits):
            #if qubit label equal to i, tensor product with Proj0 = |0><0|, Proj1 = |1><1|
            if j == i:
                full_dm_0 = np.kron(full_dm_0, dm_0)
                full_dm_1 = np.kron(full_dm_1, dm_1)
            #if qubit label is not i, tensor product with Identity
            else:
                full_dm_0 = np.kron(full_dm_0, np.eye(2))
                full_dm_1 = np.kron(full_dm_1, np.eye(2))

        # apply to basis vectors
        basis = pygsti.baseobjs.basis.Basis.cast("pp", 4**num_qubits)
        mat_0 = []
        mat_1 = []
        # parallelise the most expensive step: pass the I|0><0|III measurement matrix of qubit i together with all possible basis as a pool of inputs to convert_matrix
        #with Pool(len(basis.elements)) as p:        
        #with ThreadPool(len(basis.elements)) as p:
        #    mat_0 = p.map(convert_matrix, [(full_dm_0, e) for e in basis.elements])
        #    mat_1 = p.map(convert_matrix, [(full_dm_1, e) for e in basis.elements]) # maybe the basis means density matrix basis? --> MeM^dag gives the projected e'
        
        max_threads = os.cpu_count() - 2
        basis_lists = [basis.elements[x:x+max_threads] for x in range(0, len(basis.elements), max_threads)]
        print(len(basis_lists))
        for k in range(len(basis_lists)):
            print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix, [(full_dm_0, e) for e in basis_lists[k]]):
                    mat_0.append(result)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix, [(full_dm_1, e) for e in basis_lists[k]]):
                    mat_1.append(result)
                
        # reshape into a matrix operating in ppspace
        mat_0 = np.reshape(mat_0, (4**num_qubits, 4**num_qubits)) # total of 4^n basis for each 4^n-element matrices
        mat_1 = np.reshape(mat_1, (4**num_qubits, 4**num_qubits)) 
        #expect
        #[[M^\sigma_1]_sigma_1,..., M^\sigma_1]_sigma_n],
        # [M^\sigma_2]_sigma_1,..., M^\sigma_2]_sigma_n],
        #                      ...
        # [M^\sigma_n]_sigma_1,..., M^\sigma_n]_sigma_n]]
        matrices = {0: mat_0, 1: mat_1}
        measurement_matrices.append(matrices)
    with open(
        f"./precomputed_matrices/precomputed_matrices_{num_qubits}.pkl", "wb"
    ) as f:
        pickle.dump(measurement_matrices, f)


def load_projector_matrices_pp(file, labels):
    with open(file, "rb") as f:
        return {k: v for k, v in zip(labels, pickle.load(f))}


def prep_damping_matrices_pp(num_qubits, p_damping, truncate_order=None):
    E0 = np.array([[1,0],[0,np.sqrt(1-p_damping)]])
    E1 = np.array([[0,0],[np.sqrt(p_damping),0]])
    
    
    basis = pygsti.baseobjs.basis.Basis.cast("pp", 4**num_qubits)
    max_threads = os.cpu_count() - 2
    basis_lists = [basis.elements[x:x+max_threads] for x in range(0, len(basis.elements), max_threads)]
    
    mat_pp_total = np.zeros((4**num_qubits, 4**num_qubits))
    for i in range(2**num_qubits):
        # If only up to second order:
        if truncate_order is not None:
            if hamming_weight(i) > truncate_order:
                continue
        mat = np.array([1])
        #if hamming_weight(i) > 2 or hamming_weight(i) == 0:
        #    continue
        for j in range(num_qubits):
            if (i & (1 << j)) >> j:
                mat = np.kron(mat, E1)
            else:
                mat = np.kron(mat, E0)
        
        mat_pp = []
        for k in range(len(basis_lists)):
            print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix_hermitian, [(mat, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        
        mat_pp_total += mat_pp
    
    #p_zero = 1-(1-p_damping)^(1/num_qubits)
    #zero_mapping_operator = 
    #mat_pp_total
    
    with open(
        f"./damping_matrices/{num_qubits}_qubits_{p_damping}_truncate_{truncate_order}.pkl", "wb"
    ) as f:
        pickle.dump(mat_pp_total, f)


def prep_damping_matrices_pp_from_lindbladian(num_qubits, p_damping):
    #Y = pauli['Y']
    #I = pauli['I']
    #X = pauli['X']
    I = array([[1, 0], [0, 1]])
    X = array([[0, 1], [1, 0]])
    Y = array([[0, -1j], [1j, 0]])
    Z = array([[1, 0], [0, -1]])
    basis = pygsti.baseobjs.basis.Basis.cast("pp", 4**num_qubits)
    max_threads = os.cpu_count() - 2
    basis_lists = [basis.elements[x:x+max_threads] for x in range(0, len(basis.elements), max_threads)]
    
    mat_pp_total = np.zeros((4**num_qubits, 4**num_qubits), dtype=complex)
    
    for i in range(num_qubits):
        mat_pp_i = np.zeros((4**num_qubits, 4**num_qubits), dtype=complex)
        
        matX = np.array([1])
        for j in range(num_qubits):
            if j == i:
                matX = np.kron(matX, X)
            else:
                matX = np.kron(matX, I)
        
        matY = np.array([1])
        for j in range(num_qubits):
            if j == i:
                matY = np.kron(matY, Y)
            else:
                matY = np.kron(matY, I)       

        matZ = np.array([1])
        for j in range(num_qubits):
            if j == i:
                matZ = np.kron(matZ, Z)
            else:
                matZ = np.kron(matZ, I)
                
        matI = np.eye(2**num_qubits)
        #matXY_commutator = matX@matY - matY@matX
                
        #SX channel      
        mat_pp = []
        for k in range(len(basis_lists)):
            #print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix, [(matX, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        mat_pp_i += mat_pp
        
        #SY channel
        mat_pp = []
        for k in range(len(basis_lists)):
            #print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix, [(matY, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        mat_pp_i += mat_pp
        
        #Identities from SX and SY channel
        mat_pp = []
        for k in range(len(basis_lists)):
            #print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix, [(np.sqrt(2)*matI, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        mat_pp_i -= mat_pp
        
        #AXY channel
        #-iX \rho Y
        mat_pp = []
        for k in range(len(basis_lists)):
            #print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix_asymmetry, [(matX, matY, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        mat_pp_i -= 1j*mat_pp
        
        #+iY \rho X
        mat_pp = []
        for k in range(len(basis_lists)):
            #print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix_asymmetry, [(matY, matX, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        mat_pp_i += 1j*mat_pp
        
        #-i/2 [X,Y]\rho
        mat_pp = []
        for k in range(len(basis_lists)):
            #print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix_left, [(1j*matZ, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        mat_pp_i -= 1j*mat_pp

        #-i/2 \rho [X,Y]
        mat_pp = []
        for k in range(len(basis_lists)):
            #print(k)
            with Pool(max_threads) as p:
                for result in p.map(convert_matrix_right, [(1j*matZ, e) for e in basis_lists[k]]):
                    mat_pp.append(result)
        mat_pp = np.reshape(mat_pp, (4**num_qubits, 4**num_qubits))
        mat_pp_i -= 1j*mat_pp
        #print(mat_pp_i)
        mat_pp_total += mat_pp_i
    
    mat_pp_total = np.real(mat_pp_total)
    generator_coeff = -1/4*np.log(1-p_damping)
    mat_pp_total *= generator_coeff
    mat_pp_total = expm(mat_pp_total)
    #print(mat_pp_total)
    #p_zero = 1-(1-p_damping)^(1/num_qubits)
    #zero_mapping_operator = 
    #mat_pp_total
    
    with open(
        f"./damping_matrices/{num_qubits}_qubits_{p_damping}_from_lindbladian.pkl", "wb"
    ) as f:
        pickle.dump(mat_pp_total, f)
        
        
def load_damping_matrices_pp(file):
    with open(file, "rb") as f:
        return pickle.load(f)
    
def gen_idle_model(labels_with_noise, labels_full, p_phase, p_damping, ZZ_rate, crosstalk_axes_dict, ZZ_couplings=None):
    # Generate error dict for IDLE noise (ZZ coherent rotation + uniform Pauli channel/depolarising noise)
    noise_dict = {}
    
    idle_gate_key = ("Gi", labels_with_noise[0])
    noise_dict[idle_gate_key] = {}
    
    if ZZ_couplings is None:
        ZZ_couplings = [(i,j) for i,j in zip(labels_with_noise, labels_with_noise[1:])]
    
    #case 1: lindblad generators for single qubit covers all higher orders
    #p_qubit = 1-(1-p)**(1/len(labels_with_noise))   # (1-p_qubit)^N = 1 - P_total, here the input p is the total probability
    #for q in labels_with_noise:
    #    noise_dict[idle_gate_key][f'SX:{q}'] = p_qubit/3
    #    noise_dict[idle_gate_key][f'SY:{q}'] = p_qubit/3
    #    noise_dict[idle_gate_key][f'SZ:{q}'] = p_qubit/3
    
    
    #case 2: Must explicitly express higher order lindbald generators
    #p_qubit_noise = 1-(1-p)**(1/len(labels_with_noise))
    #p_qubit_id = 1 - p_qubit_noise
    #for order in range(1, len(labels_with_noise)+1):
    #    pauli_channels_basis = [['X','Y','Z'] for i in range(order)]
    #    pauli_channels = []
    #    for basis in itertools.product(*pauli_channels_basis):
    #        pauli_channels.append(''.join(basis))
        
    #    target_qubits_basis = [[q for q in labels_with_noise] for i in range(order)]
    #    target_qubits = []
    #    for basis in itertools.product(*target_qubits_basis):
    #        if all(i < j for i, j in zip(basis, basis[1:])):
    #            target_qubits_string = ''
    #            for q in basis:
    #                target_qubits_string += f'{q},'
    #            target_qubits.append(target_qubits_string[:-1])
        
    #    for targets in target_qubits:
    #        for pauli in pauli_channels:
    #            noise_dict[idle_gate_key]['S'+pauli+':'+targets] = (p_qubit_noise/3)**order
    
    #case 3: Dephasing noise only
    #p_qubit_noise = 1-(1-p_phase)**(1/len(labels_with_noise))
    #for order in range(1, len(labels_with_noise)+1):
    #    pauli_basis = 'Z'*order
        
    #    target_qubits_basis = [[q for q in labels_with_noise] for i in range(order)]
    #    target_qubits = []
    #    for basis in itertools.product(*target_qubits_basis):
    #        if all(i < j for i, j in zip(basis, basis[1:])):
    #            target_qubits_string = ''
    #            for q in basis:
    #                target_qubits_string += f'{q},'
    #            target_qubits.append(target_qubits_string[:-1])
        
    #    for targets in target_qubits:
    #        noise_dict[idle_gate_key]['S'+pauli_basis+':'+targets] = (p_qubit_noise)**order
    
    #case 4: Dephasing noise only but p is the qubit noise probability (instead of total probability)
    #for order in range(1, len(labels_with_noise)+1):
    #    pauli_basis = 'Z'*order
        
    #    target_qubits_basis = [[q for q in labels_with_noise] for i in range(order)]
    #    target_qubits = []
    #    for basis in itertools.product(*target_qubits_basis):
    #        if all(i < j for i, j in zip(basis, basis[1:])):
    #            target_qubits_string = ''
    #            for q in basis:
    #                target_qubits_string += f'{q},'
    #            target_qubits.append(target_qubits_string[:-1])
        
    #    for targets in target_qubits:
    #        noise_dict[idle_gate_key]['S'+pauli_basis+':'+targets] = (p_phase)**order
    
    #case 5: Dephasing noise only but p is the qubit noise probability (instead of total probability), and highest order is 2
    #for order in range(1, min(2,len(labels_with_noise))+1):
    #    lindblad_coeff = np.log(1-2*p_phase)/(-2)
    #    
    #    
    #    pauli_basis = 'Z'*order
    #    target_qubits_basis = [[q for q in labels_with_noise] for i in range(order)]
    #    target_qubits = []
    #    for basis in itertools.product(*target_qubits_basis):
    #        if all(i < j for i, j in zip(basis, basis[1:])):
    #            target_qubits_string = ''
    #            for q in basis:
    #                target_qubits_string += f'{q},'
    #            target_qubits.append(target_qubits_string[:-1])
    #    
    #    for targets in target_qubits:
    #        noise_dict[idle_gate_key]['S'+pauli_basis+':'+targets] = (p_phase)**order
    #        noise_dict[idle_gate_key]['S'+pauli_basis+':'+targets] = lindblad_coeff
    
    
    lindblad_coeff = np.log(1-2*p_phase)/(-2)
    pauli_basis = 'Z'
    
    for qubit in labels_with_noise:
        noise_dict[idle_gate_key][f'S{pauli_basis}:{qubit}'] = lindblad_coeff
        
        
    #Generate Affine/Damping noise towards 0:
    #for order in range(1, min(1,len(labels_with_noise))+1):
    #    pauli_basis = 'Z'*order
    #    
    #    target_qubits_basis = [[q for q in labels_with_noise] for i in range(order)]
    #    target_qubits = []
    #    for basis in itertools.product(*target_qubits_basis):
    #        if all(i < j for i, j in zip(basis, basis[1:])):
    #            target_qubits_string = ''
    #            for q in basis:
    #                target_qubits_string += f'{q},'
    #            target_qubits.append(target_qubits_string[:-1])
    #    
    #    for targets in target_qubits:
    #        noise_dict[idle_gate_key]['A'+pauli_basis+':'+targets] = (p_damping)**order
    
    #Generate Affine/Damping noise towards 0: (basis fixed (?))
    #for order in range(1, min(2,len(labels_with_noise))+1):      
    #    pauli_basis = [['X','Y'] for i in range(order)]
    #    target_paulis = []
    #    for basis in itertools.product(*pauli_basis):
    #        pauli_basis_string = ''
    #        for pauli in basis:
    #            pauli_basis_string += pauli
    #        target_paulis.append(pauli_basis_string)
    #    
    #    target_qubits_basis = [[q for q in labels_with_noise] for i in range(order)]
    #    target_qubits = []
    #    for basis in itertools.product(*target_qubits_basis):
    #        if all(i < j for i, j in zip(basis, basis[1:])):
    #            target_qubits_string = ''
    #            for q in basis:
    #                target_qubits_string += f'{q},'
    #            target_qubits.append(target_qubits_string[:-1])
    #    
    #    for basis in target_paulis:
    #        for targets in target_qubits:
    #            noise_dict[idle_gate_key]['A'+basis+':'+targets] = (p_damping)**order
    
    #Generate always-on ZZ noise
    for pair in ZZ_couplings:
        noise_dict[idle_gate_key]['HZZ:'+f'{pair[0]},{pair[1]}'] = ZZ_rate
    
    
    # Generate error dict for crosstalk noise (random single qubit rotation)
    #attack_qubits = sorted(set(labels_full) - set(labels_with_noise))
    crosstalk_gate_key = ("Gcrosstalk", labels_with_noise[0])
    noise_dict[crosstalk_gate_key] = {}
    
    for target, axis in crosstalk_axes_dict.items():
        if isinstance(target, int):
            axis_normalised = np.array(axis[0])/norm(axis[0])
            rotation_params = axis[1]/2*axis_normalised
            noise_dict[crosstalk_gate_key][f'HX:{target}'] = rotation_params[0]
            noise_dict[crosstalk_gate_key][f'HY:{target}'] = rotation_params[1]
            noise_dict[crosstalk_gate_key][f'HZ:{target}'] = rotation_params[2]
        
        elif isinstance(target, tuple):
            axis_normalised = np.array(axis[0])/norm(axis[0])
            rotation_params = axis[1]/2*axis_normalised
            target_formatted = f'{target[0]},{target[1]}'
            noise_dict[crosstalk_gate_key][f'HXX:{target_formatted}'] = rotation_params[0]
            noise_dict[crosstalk_gate_key][f'HXY:{target_formatted}'] = rotation_params[1]
            noise_dict[crosstalk_gate_key][f'HXZ:{target_formatted}'] = rotation_params[2]
            noise_dict[crosstalk_gate_key][f'HYX:{target_formatted}'] = rotation_params[3]
            noise_dict[crosstalk_gate_key][f'HYY:{target_formatted}'] = rotation_params[4]
            noise_dict[crosstalk_gate_key][f'HYZ:{target_formatted}'] = rotation_params[5]
            noise_dict[crosstalk_gate_key][f'HZX:{target_formatted}'] = rotation_params[6]
            noise_dict[crosstalk_gate_key][f'HZY:{target_formatted}'] = rotation_params[7]
            noise_dict[crosstalk_gate_key][f'HZZ:{target_formatted}'] = rotation_params[8]
            
    #print(noise_dict)
    return QuantumSim(labels_with_noise, noise_dict, p_damping=p_damping)



class QuantumSim:
    def __init__(self, labels, error_dict, p_damping):
        #allowed basis gates/directions settings initialize
        pspec = pygsti.processors.QubitProcessorSpec(
            num_qubits=len(labels),
            qubit_labels=labels,
            gate_names=[
                "Gxpi",
                "Gypi",
                "Gzpi",
                "Gx",
                "Gy",
                "Gz",
                "Gh",
                "Gp",
                "Gt",
                "Gtdg",
                "Gzr",
                "Gcnot",
                "Gcnot_ideal",
                "Gcnot2",
                "Gcphase",
                "Gswap",
                "Gsdg",
                "Gzpi10",
                "Gzpi40",
                "Gzpi80",
                "Gzpi160",
                "Gz-pi10",
                "Gz-pi40",
                "Gz-pi80",
                "Gz-pi160",
                "Gypi10",
                "Gypi40",
                "Gypi80",
                "Gypi160",
                "Gy-pi10",
                "Gy-pi40",
                "Gy-pi80",
                "Gy-pi160",
                "Gi",
                "Gcrosstalk"
            ],
            nonstd_gate_unitaries={
                "Gsdg": np.array([[1, 0], [0, -1j]], dtype=complex),
                "Gtdg": np.array([[1, 0], [0, np.exp(-1j*pi/4)]], dtype=complex),
                "Gzpi10": np.array([[1, 0], [0, np.exp(1j*pi/10)]], dtype=complex),
                "Gzpi40": np.array([[1, 0], [0, np.exp(1j*pi/40)]], dtype=complex),
                "Gzpi80": np.array([[1, 0], [0, np.exp(1j*pi/80)]], dtype=complex),
                "Gzpi160": np.array([[1, 0], [0, np.exp(1j*pi/160)]], dtype=complex),
                "Gz-pi10": np.array([[1, 0], [0, np.exp(-1j*pi/10)]], dtype=complex),
                "Gz-pi40": np.array([[1,0],[0,np.exp(-1j*pi/40)]], dtype=complex),
                "Gz-pi80": np.array([[1,0],[0,np.exp(-1j*pi/80)]], dtype=complex),
                "Gz-pi160": np.array([[1,0],[0,np.exp(-1j*pi/160)]], dtype=complex),
                "Gypi10": np.array([[cos(pi/20),-sin(pi/20)],[sin(pi/20),cos(pi/20)]], dtype=complex),
                "Gypi40": np.array([[cos(pi/80),-sin(pi/80)],[sin(pi/80),cos(pi/80)]], dtype=complex),
                "Gypi80": np.array([[cos(pi/160),-sin(pi/160)],[sin(pi/160),cos(pi/160)]], dtype=complex),
                "Gypi160": np.array([[cos(pi/320),-sin(pi/320)],[sin(pi/320),cos(pi/320)]], dtype=complex),
                "Gy-pi10": np.array([[cos(pi/20),sin(pi/20)],[-sin(pi/20),cos(pi/20)]], dtype=complex),
                "Gy-pi40": np.array([[cos(pi/80),sin(pi/80)],[-sin(pi/80),cos(pi/80)]], dtype=complex),
                "Gy-pi80": np.array([[cos(pi/160),sin(pi/160)],[-sin(pi/160),cos(pi/160)]], dtype=complex),
                "Gy-pi160": np.array([[cos(pi/320),sin(pi/320)],[-sin(pi/320),cos(pi/320)]], dtype=complex),
                "Gcnot_ideal": CNOT,
                "Gcnot2": np.identity(2**4, dtype=complex),
                "Gcrosstalk": np.array([[1, 0], [0, 1]], dtype=complex)
            },
            availability={
                "Gcnot": "all-permutations",
                "Gcnot_ideal": "all-permutations",
                "Gcnot2": "all-permutations",
                "Gcphase": "all-permutations",
                "Gswap": "all-permutations",
            },
        )  # empty dict specifies a fully connected device?


                
        self.error_dict = error_dict
        self.mdl = pygsti.models.create_cloud_crosstalk_model(pspec, lindblad_error_coeffs=error_dict)#,
                                                              #depolarization_strengths={'Gi':0.01},
                                                              #depolarization_parameterization='depolarize')#crosstalk error model should be included in new_error_dict
        #self.mdl_trivial = pygsti.models.create_explicit_model(pspec, embed_gates=True)
        #self.mdl.create_operation()
        self.rho = self.mdl.prep_blks["layers"]["rho0"]
        self.num_qubits = len(labels)
        self.labels = labels
        try:
            self.pp_id = pygsti.tools.stdmx_to_ppvec(np.eye(2**self.num_qubits))
        except:
            warnings.warn('number of qubits exceeded maximum matrix dimension, pp_id attribute not established')
        try:
            self.measurement_matrices = load_projector_matrices_pp(
            f"./precomputed_matrices/precomputed_matrices_{self.num_qubits}.pkl",
            self.labels,
            )
            
        except:
            warnings.warn('number of qubits exceeded maximum precomputed matrices, measurement_matrices attribute not established.')
        
        try:
            self.damping_matrix = load_damping_matrices_pp(
            f"./damping_matrices/{self.num_qubits}_qubits_{p_damping}_truncate_2.pkl"
            )
            
            #self.damping_matrix = load_damping_matrices_pp(
            #    f"./damping_matrices/{self.num_qubits}_qubits_{p_damping}_from_lindbladian.pkl"
            #)
            
        except:
            warnings.warn('number of qubits exceeded maximum damping matrix, damping_matrix attribute not established.')
            
        self.rand_gates = {q:[] for q in labels}|{(q1, q2): [] for q1 in labels for q2 in labels if q1<q2}#|{(q1, q2, q3): [] for q1 in labels for q2 in labels for q3 in labels if q1<q2<q3}
        self.rand_gate_names = {q:[] for q in labels}|{(q1, q2): [] for q1 in labels for q2 in labels if q1<q2}
    
    #def test(self):
    #    print('hi')
    
    def idle(self):
        if self.mdl.operation_blks["cloudnoise"].get(("Gi", self.labels[0])) is not None:
            self.rho = self.mdl.operation_blks["cloudnoise"][("Gi", self.labels[0])].acton(
                self.rho
                )
        #self.rho = self.mdl.operation_blks['layers']['Gi',self.labels[0]].acton(self.rho)
        
            rho = self.rho
            rho = self.damping_matrix @ rho
            rho = rho / np.dot(np.real(self.pp_id.T), rho)  # re-normalize
            self.rho = pygsti.modelmembers.states.StaticState(rho)
            
        #self.rho = self.mdl.operation_blks["layers"][("Gi", self.labels[0])].acton(self.rho)
        
    def crosstalk(self):
        if self.mdl.operation_blks["cloudnoise"].get(("Gcrosstalk", self.labels[0])) is not None:
            self.rho = self.mdl.operation_blks["cloudnoise"][("Gcrosstalk", self.labels[0])].acton(
                self.rho
                )
        
        
    def h(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gh", q)].acton(self.rho)

    def x(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gxpi", q)].acton(self.rho)
    
    def xpi2(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gx", q)].acton(self.rho)

    def y(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gypi", q)].acton(self.rho)

    def z(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gzpi", q)].acton(self.rho)

    def s(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gp", q)].acton(self.rho)
        
    def sdg(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gsdg", q)].acton(self.rho)
        
    def t(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gt", q)].acton(self.rho)
    
    def tdg(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gtdg", q)].acton(self.rho)
    
    def zpi10(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gzpi10", q)].acton(self.rho)
        
    def zpi40(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gzpi40", q)].acton(self.rho)
        
    def zpi80(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gzpi80", q)].acton(self.rho)
    
    def zpi160(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gzpi160", q)].acton(self.rho)
    
    def zminuspi10(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gz-pi10", q)].acton(self.rho)
        
    def zminuspi40(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gz-pi40", q)].acton(self.rho)
    
    def zminuspi80(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gz-pi80", q)].acton(self.rho)
    
    def zminuspi160(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gz-pi160", q)].acton(self.rho)
    
    def ypi10(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gypi10", q)].acton(self.rho)
        
    def ypi40(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gypi40", q)].acton(self.rho)
    
    def ypi80(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gypi80", q)].acton(self.rho)
    
    def ypi160(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gypi160", q)].acton(self.rho)
    
    def yminspi10(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gy-pi10", q)].acton(self.rho)
        
    def yminspi40(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gy-pi40", q)].acton(self.rho)
    
    def yminspi80(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gy-pi80", q)].acton(self.rho)
    
    def yminspi160(self, q):
        self.rho = self.mdl.operation_blks["layers"][("Gy-pi160", q)].acton(self.rho)
    
    def cnot(self, c, t):
        #cnt with crosstalk noise, but first check if the dedicated CNOT operation does induce crosstalk in cloudnoise
        if self.mdl.operation_blks["cloudnoise"].get(("Gcnot", c, t)) is not None:
            self.rho = self.mdl.operation_blks["cloudnoise"][("Gcnot", c, t)].acton(
                self.rho
            ) # if does, add the error gate to rho
        self.rho = self.mdl.operation_blks["layers"][("Gcnot", c, t)].acton(self.rho)
        
    def cnot_Xtalk_only(self, c, t):
        
        if self.mdl.operation_blks["cloudnoise"].get(("Gcnot", c, t)) is not None:
            self.rho = self.mdl.operation_blks["cloudnoise"][("Gcnot", c, t)].acton(
                self.rho
            )
        
    def cnotn(self, c, t):  # cnot with no cross talk noise
        # self.rho = self.mdl.operation_blks['cloudnoise'][('Gcnot', c, t)].acton(self.rho)
        self.rho = self.mdl.operation_blks["layers"][("Gcnot", c, t)].acton(self.rho)
    
    def two_cnots(self, pair1, pair2):
        (c1, t1) = pair1
        (c2, t2) = pair2
        if self.mdl.operation_blks["cloudnoise"].get(("Gcnot2", c1, t1, c2, t2)) is not None:
            self.rho = self.mdl.operation_blks["cloudnoise"][("Gcnot2", c1, t1, c2, t2)].acton(self.rho)
        
        self.rho = self.mdl.operation_blks["layers"][("Gcnot", c1, t1)].acton(self.rho)
        self.rho = self.mdl.operation_blks["layers"][("Gcnot", c2, t2)].acton(self.rho)
        
    def ccz_cx_decomposed(self, c1, c2, t, vaccine=False, vaccine_params=None, noise=True):
        """apply CNOTs decomposed CCZ gate to c1, c2, t

        Args:
            c1 (int): control 1 qubit label
            c2 (int): control 2 qubit label
            t (int): target qubit label
            vaccine (bool, optional): whether we apply correction to crosstalk noises. Defaults to False.
            vaccine_params (dict, optional): dictionary of vaccine parameters, keys are CX source, values are dicts of error params on
            different qubits. Defaults to None.
            noise (bool, optional): whether crosstalk noise are activated. Defaults to True.

        Raises:
            ValueError: occurs if noise is False and vaccine is True
        """
        
        if (noise is False) and (vaccine is True):
            #warnings.warn('noise is set to False but vaccine is set to True, double check your setting', stacklevel=2)
            raise ValueError('noise is set to False but vaccine is set to True, double check your setting')
        
        self.rand_gates[(c1, c2, t)].append(CCZ)
        
        if noise is False:
            self.cnotn(c2, t)
            self.tdg(t)
            #CNOT(c1, t) by swapping c2 and t first (not using Gswap here because we want it to fully decompose into CNOTs)
            self.cnotn(t, c2)
            self.cnotn(c2, t)
            self.cnotn(t, c2)
            self.cnotn(c1, c2)
            self.cnotn(t, c2) #SWAP back
            self.cnotn(c2, t)
            self.cnotn(t, c2)
            
            self.t(t)
            self.cnotn(c2, t)
            self.tdg(t)
            
            #CNOT(c1, t) by swapping again
            self.cnotn(t, c2)
            self.cnotn(c2, t)
            self.cnotn(t, c2)
            self.cnotn(c1, c2)
            self.cnotn(t, c2)
            self.cnotn(c2, t)
            self.cnotn(t, c2)

            self.t(c2)
            self.t(t)
            self.cnotn(c1, c2)
            self.t(c1)
            self.tdg(c2)
            self.cnotn(c1, c2)
            
        else:
            if vaccine is False:
                self.cnot(c2, t)
                self.tdg(t)
                #CNOT(c1, t) by swapping c2 and t first (not using Gswap here because we want it to fully decompose into CNOTs)
                self.cnot(t, c2)
                self.cnot(c2, t)
                self.cnot(t, c2)
                self.cnot(c1, c2)
                self.cnot(t, c2) #SWAP back
                self.cnot(c2, t)
                self.cnot(t, c2)
            
                self.t(t)
                self.cnot(c2, t)
                self.tdg(t)
            
                #CNOT(c1, t) by swapping again
                self.cnot(t, c2)
                self.cnot(c2, t)
                self.cnot(t, c2)
                self.cnot(c1, c2)
                self.cnot(t, c2)
                self.cnot(c2, t)
                self.cnot(t, c2)

                self.t(c2)
                self.t(t)
                self.cnot(c1, c2)
                self.t(c1)
                self.tdg(c2)
                self.cnot(c1, c2)
            
            else:
                rotation_c2_t = QuantumSim.euler_angles_from_axis_angle(vaccine_params[(c2, t)][c1][0],
                                                                        vaccine_params[(c2, t)][c1][1])
                rotation_t_c2 = QuantumSim.euler_angles_from_axis_angle(vaccine_params[(t, c2)][c1][0],
                                                                        vaccine_params[(t, c2)][c1][1])
                rotation_c1_c2 = QuantumSim.euler_angles_from_axis_angle(vaccine_params[(c1, c2)][t][0],
                                                                         vaccine_params[(c1, c2)][t][1])
                
                self.cnot(c2, t)
                self.apply_correction(c1, rotation_c2_t)
                
                self.tdg(t)
                
                #CNOT(c1, t) by swapping c2 and t first (not using Gswap here because we want it to fully decompose into CNOTs)
                self.cnot(t, c2)
                self.apply_correction(c1, rotation_t_c2)
                
                self.cnot(c2, t)
                self.apply_correction(c1, rotation_c2_t)
                
                self.cnot(t, c2)
                self.apply_correction(c1, rotation_t_c2)
                
                self.cnot(c1, c2)
                self.apply_correction(t, rotation_c1_c2)
                
                self.cnot(t, c2) #SWAP back
                self.apply_correction(c1, rotation_t_c2)
                
                self.cnot(c2, t)
                self.apply_correction(c1, rotation_c2_t)
                
                self.cnot(t, c2)
                self.apply_correction(c1, rotation_t_c2)
            
                self.t(t)
                
                self.cnot(c2, t)
                self.apply_correction(c1, rotation_c2_t)
                
                self.tdg(t)
            
                #CNOT(c1, t) by swapping again
                self.cnot(t, c2)
                self.apply_correction(c1, rotation_t_c2)
                
                self.cnot(c2, t)
                self.apply_correction(c1, rotation_c2_t)
                
                self.cnot(t, c2)
                self.apply_correction(c1, rotation_t_c2)
                
                self.cnot(c1, c2)
                self.apply_correction(t, rotation_c1_c2)
                
                self.cnot(t, c2)
                self.apply_correction(c1, rotation_t_c2)
                
                self.cnot(c2, t)
                self.apply_correction(c1, rotation_c2_t)
                
                self.cnot(t, c2)
                self.apply_correction(c1, rotation_t_c2)

                self.t(c2)
                self.t(t)
                self.cnot(c1, c2)
                self.apply_correction(t, rotation_c1_c2)
                self.t(c1)
                self.tdg(c2)
                self.cnot(c1, c2)
                self.apply_correction(t, rotation_c1_c2)
                
                
    def random_evolution(self, q):
        a = random.random()
        if 0 < a < 1/3:
            self.h(q)
            self.rand_gates[q].append(H)
        elif 1/3 < a < 2/3:
            self.t(q)
            self.rand_gates[q].append(T)
        else:
            self.x(q)
            self.rand_gates[q].append(X)
    
    def random_evolution_2q(self, q1, q2):
        a = random.random()
        b = random.random()
        if 0 < a < 1/4:
            if 0 < b < 1/2:
                #self.cnot(q1, q2)
                self.cnotn(q1, q2)
                self.rand_gates[(q1, q2)].append(CNOT)
                self.rand_gate_names[(q1, q2)].append(('CNOT', (q1, q2)))
            else:
                self.cnotn(q2, q1)
                self.rand_gates[(q1, q2)].append(CNOT_reserved)
                self.rand_gate_names[(q1, q2)].append(('CNOT', (q2, q1)))
        else:
            if 0 < b < 1/2:
                if 1/4 < a < 1/2:
                    self.h(q1)
                    self.rand_gates[(q1, q2)].append(np.kron(H, pauli['I']))
                    self.rand_gate_names[(q1, q2)].append(('H', (q1,)))
                
                elif 1/2 < a < 3/4:
                    self.s(q1)
                    self.rand_gates[(q1, q2)].append(np.kron(S, pauli['I']))
                    self.rand_gate_names[(q1, q2)].append(('S', (q1,)))

                else:
                    self.t(q1)
                    self.rand_gates[(q1, q2)].append(np.kron(T, pauli['I']))
                    self.rand_gate_names[(q1, q2)].append(('T', (q1,)))
            
            else:
                if 1/4 < a < 1/2:
                    self.h(q2)
                    self.rand_gates[(q1, q2)].append(np.kron(pauli['I'], H))
                    self.rand_gate_names[(q1, q2)].append(('H', (q2,)))
                
                elif 1/2 < a < 3/4:
                    self.s(q2)
                    self.rand_gates[(q1, q2)].append(np.kron(pauli['I'], S))
                    self.rand_gate_names[(q1, q2)].append(('S', (q2,)))

                else:
                    self.t(q2)
                    self.rand_gates[(q1, q2)].append(np.kron(pauli['I'], T))
                    self.rand_gate_names[(q1, q2)].append(('T', (q2,)))

        return 
    
    def Grover_evolution_3q(self, q1, q2, q3, oracle='101', vaccine=False, vaccine_params=None, noise=True):
        qubit_orders = (q1, q2, q3)
        if (noise is False) and (vaccine is True):
            #warnings.warn('noise is set to False but vaccine is set to True, double check your setting', stacklevel=2)
            raise ValueError('noise is set to False but vaccine is set to True, double check your setting')
        
        #Oracle
        for i in range(len(oracle)):
            if oracle[i] == '0':
                self.x(qubit_orders[i])
        self.rand_gates[(q1, q2, q3)].append(ft.reduce(np.kron, [pauli['X'] if oracle[j]=='0' else pauli['I'] for j in range(len(oracle))]))
        #CCZ decomposed into CNOTs and Ts
        self.ccz_cx_decomposed(q1, q2, q3, vaccine=vaccine, vaccine_params=vaccine_params, noise=noise)
        
        for i in range(len(oracle)):
            if oracle[i] == '0':
                self.x(qubit_orders[i])
        self.rand_gates[(q1, q2, q3)].append(ft.reduce(np.kron, [pauli['X'] if oracle[j]=='0' else pauli['I'] for j in range(len(oracle))]))
        
        #Diffusion
        for i in range(len(oracle)):
            self.h(qubit_orders[i])
        self.rand_gates[(q1, q2, q3)].append(ft.reduce(np.kron, [H, H, H]))
        
        for i in range(len(oracle)):
            self.x(qubit_orders[i])
        self.rand_gates[(q1, q2, q3)].append(ft.reduce(np.kron, [pauli['X'], pauli['X'], pauli['X']]))
        
        self.ccz_cx_decomposed(q1, q2, q3, vaccine=vaccine, vaccine_params=vaccine_params, noise=noise)
        
        for i in range(len(oracle)):
            self.x(qubit_orders[i])
        self.rand_gates[(q1, q2, q3)].append(ft.reduce(np.kron, [pauli['X'], pauli['X'], pauli['X']]))
        
        for i in range(len(oracle)):
            self.h(qubit_orders[i])
        self.rand_gates[(q1, q2, q3)].append(ft.reduce(np.kron, [H, H, H]))
        
    #def measure(self, q):
    #    full_dm_0 = self.measurement_matrices[q][0]
    #    full_dm_1 = self.measurement_matrices[q][1]
    #    rho = self.rho
    #    prob_0 = np.dot((full_dm_0 @ rho).T, self.pp_id)
    #    # Note since the matrix full_dm is always symmetric, hence equivalent to its transpose
    #    #[[M^\sigma_1]_sigma_1,..., M^\sigma_n]_sigma_1],
    #    # [M^\sigma_1]_sigma_2,..., M^\sigma_n]_sigma_2],
    #    #                      ...
    #    # [M^\sigma_1]_sigma_n,..., M^\sigma_n]_sigma_n]]
    #    # Therefore, full_dm_0 @ rho gives the projected rho in the pp rep.
    #    # then dot with self.pp_id which always equal to [2^(n/2),0,0.....], giving the factor of (I*I...*I)/(2^(n/2)) decomposition (elements on diagonals).
    #    # prob0 = tr(\rho_projected) = tr(factor*(I*I...*I)/(2^(n/2)))+tr(other paulis basis) = tr(factor*(I*I...*I)/(2^(n/2))) = tr(2^(n/2)*factor*(I*I...*I)/(2^n))=2^(n/2)*factor
    #    
    #    
    #    # print(prob_0)
#
    #    outcome = 0 if random.random() < prob_0 else 1
    #    if outcome == 0:
    #        rho = full_dm_0 @ rho
    #    else:
    #        rho = full_dm_1 @ rho
    #    rho = rho / np.dot(self.pp_id.T, rho)  # re-normalize
    #    # print(np.shape(rho))
    #    self.rho = pygsti.modelmembers.states.StaticState(rho)
    #    return outcome

    #def measure_single(self, q):
    #    computational_povm = pygsti.modelmembers.povms.ComputationalBasisPOVM(nqubits=1)
    #    self.mdl.povm_blks['Mdefault'] = computational_povm # error msg: TypeError: 'CloudNoiseModel' object does not support item assignment
    #    self.rho = self.mdl.povm_blks["Mdefault"][q].acton(self.rho)

    def prob0(self, q):
        full_dm_0 = self.measurement_matrices[q][0]
        #full_dm_1 = self.measurement_matrices[q][1]
        rho = self.rho
        prob_0 = np.dot((full_dm_0 @ rho).T, self.pp_id)[0][0]
        return prob_0
    
    def prob1(self, q):
        full_dm_0 = self.measurement_matrices[q][0]
        #full_dm_1 = self.measurement_matrices[q][1]
        rho = self.rho
        prob_0 = np.dot((full_dm_0 @ rho).T, self.pp_id)[0][0]
        return 1-prob_0
        
        
    def get_partial_trace(self, trace):
        # partial trace to get the reduced density matrix by tracing over the qubits in [trace]
        keep = [x for x in self.labels if x not in trace] # qubit labels don't trace out
        keep_idx = [self.labels.index(i) for i in keep] # qubit label positions in labels don't trace out
        keep = np.asarray(keep)
        dims = 2 * np.ones(self.num_qubits, int) #n lost of [2, 2, 2, ... , 2]
        Ndim = dims.size #Ndim = N

        idx1 = [i for i in range(Ndim)] #[0,1,2,3...,n-1]
        idx2 = [Ndim + i if i in keep_idx else i for i in range(Ndim)] # e.g. [0,1,2+n (2 is in keep_idx),3,...n-1]
        rho = pygsti.tools.ppvec_to_stdmx(self.rho) #convert back to standard matrix rep (2^n, 2^n) (total 2^2n elements)
        rho_a = rho.reshape(np.tile(dims, 2)) # decompose rho from pp column vec the dimension [2,2,...,2] with 2n lost of twos, therefore 2^2n elements)
        rho_a = np.einsum(rho_a, idx1 + idx2) # idx1+idx2 = [0,1,2,3...,n-1, 0,1,2+n (2 is in keep_idx),3,...n-1] (the only non-repetting indicies are 2&2+n, so they are non-traced,
                                              # otherwise, einstein sum all indicies on 0,0 and 1,1 and 3,3 ....)
        return rho_a.reshape(2**keep.size, 2**keep.size)

    #def reset(self, q):
    #    outcome = self.measure(q)
    #    if outcome == 1:
    #        self.x(q)
    #    return outcome
    #
    #def reset0_only(self, q):
    #    rho = self.rho
    #    full_dm_0 = self.self.measurement_matrices[q][0]
    #    
    #    rho = full_dm_0 @ rho
    #    rho = rho / np.dot(self.pp_id.T, rho)
    #    self.rho = pygsti.modelmembers.states.StaticState(rho)
    
    def Xtalk_vaccine_tomography_fast(self, attack_origins=[(13,12)], vaccine_qubits=[14],
                                      cnot_count1=2, cnot_count2=4):
        pauli_basis_exepected_values = {pair: {q: [{'X':0, 'Y':0, 'Z':0} 
                                                   for i in range(2)] 
                                               for q in vaccine_qubits} #if q not in pair} 
                                        for pair in attack_origins}
        #cnot_counts = [cnot_count1, cnot_count2]
        
        for pair in attack_origins:
            self.rho = self.mdl.prep_blks["layers"]["rho0"]
            vaccine_qubits_for_pair = []
            for s in vaccine_qubits:
                #if s not in pair:
                vaccine_qubits_for_pair.append(s)
                
            for i in range(1, cnot_count2+1):
                #self.cnot(pair[0], pair[1])
                self.cnot_Xtalk_only(pair[0], pair[1])
                if i == cnot_count1:
                    for q in vaccine_qubits_for_pair:
                        reduced_rho = self.get_partial_trace([x for x in self.labels if x!=q])
                        pauli_basis_exepected_values[pair][q][0]['X'] = np.real(np.trace(pauli['X']@reduced_rho))
                        pauli_basis_exepected_values[pair][q][0]['Y'] = np.real(np.trace(pauli['Y']@reduced_rho))
                        pauli_basis_exepected_values[pair][q][0]['Z'] = np.real(np.trace(pauli['Z']@reduced_rho))
                        
                elif i == cnot_count2:
                    for q in vaccine_qubits_for_pair:
                        reduced_rho = self.get_partial_trace([x for x in self.labels if x!=q])
                        pauli_basis_exepected_values[pair][q][1]['X'] = np.real(np.trace(pauli['X']@reduced_rho))
                        pauli_basis_exepected_values[pair][q][1]['Y'] = np.real(np.trace(pauli['Y']@reduced_rho))
                        pauli_basis_exepected_values[pair][q][1]['Z'] = np.real(np.trace(pauli['Z']@reduced_rho))
        
        return pauli_basis_exepected_values
    
    def Xtalk_vaccine_extended(self, attack_origins=[(13,12)], vaccine_qubits=[14], 
                               cnot_count1=2, cnot_count2=4):
        
        pauli_basis_exepected_values = {pair: {q: [{'X':0, 'Y':0, 'Z':0} 
                                                   for i in range(2)] 
                                               for q in vaccine_qubits} #if q not in pair} 
                                        for pair in attack_origins}
        #cnot_counts = [cnot_count1, cnot_count2]
        
        for pair in attack_origins:
            #self.rho = self.mdl.prep_blks["layers"]["rho0"]
            circ = Circuit(line_labels=self.labels, editable=True)
            vaccine_qubits_for_pair = []
            for s in vaccine_qubits:
                #if s not in pair:
                vaccine_qubits_for_pair.append(s)
                
            for i in [cnot_count1, cnot_count2]:
                #self.cnot(pair[0], pair[1])
                #self.cnot_Xtalk_only(pair[0], pair[1])
                circ_new = circ.copy()
                for j in range(i):
                    circ_new.append_circuit_inplace(Circuit([(('Gcnot', pair[0], pair[1]))], line_labels=self.labels))
                #circ_new.done_editing()

                for q in vaccine_qubits_for_pair:
                    circ_x = circ_new.copy()
                    circ_x.append_circuit_inplace(Circuit([(('Gh', q))], line_labels=self.labels))
                    circ_x.done_editing()
                    circ_y = circ_new.copy()
                    circ_y.append_circuit_inplace(Circuit([(('Gsdg', q))], line_labels=self.labels))
                    circ_y.append_circuit_inplace(Circuit([(('Gh', q))], line_labels=self.labels))
                    circ_y.done_editing()
                    circ_z = circ_new.copy()
                    circ_z.done_editing()
                    
                    q_idx = self.labels.index(q)
                    
                    expected_X = 0
                    probabilities = self.mdl.probabilities(circ_x)
                    for outcome, prob in probabilities.items():
                        if outcome[0][q_idx] == '0':
                            expected_X += prob
                        else:
                            expected_X -= prob
                    
                    expected_Y = 0
                    probabilities = self.mdl.probabilities(circ_y)
                    for outcome, prob in probabilities.items():
                        if outcome[0][q_idx] == '0':
                            expected_Y += prob
                        else:
                            expected_Y -= prob
                            
                    expected_Z = 0
                    probabilities = self.mdl.probabilities(circ_z)
                    for outcome, prob in probabilities.items():
                        if outcome[0][q_idx] == '0':
                            expected_Z += prob
                        else:
                            expected_Z -= prob

                    if i == cnot_count1:
                        pauli_basis_exepected_values[pair][q][0]['X'] = expected_X
                        pauli_basis_exepected_values[pair][q][0]['Y'] = expected_Y
                        pauli_basis_exepected_values[pair][q][0]['Z'] = expected_Z
                        
                    elif i == cnot_count2:
                        pauli_basis_exepected_values[pair][q][1]['X'] = expected_X
                        pauli_basis_exepected_values[pair][q][1]['Y'] = expected_Y
                        pauli_basis_exepected_values[pair][q][1]['Z'] = expected_Z
        
        return pauli_basis_exepected_values
    
    
    def Xtalk_vaccine_tomography(self, attack_origins=[(13,12)], vaccine_qubits=[14], 
                                 cnot_count1=2, cnot_count2=4, sample_size=100, separate=False):
        
        pauli_basis_exepected_values = {pair: {q: [{'X':0, 'Y':0, 'Z':0} 
                                                   for i in range(2)] 
                                               for q in vaccine_qubits} #if q not in pair} 
                                        for pair in attack_origins}
        cnot_counts = [cnot_count1, cnot_count2]
        
        if separate is False:
            
            for pair in attack_origins:
                self.rho = self.mdl.prep_blks["layers"]["rho0"]

                #First point (apart from north pole |0>)
                for i in range(sample_size):
                    if i % 100 == 0:
                        print(i)
                    for j in range(2):
                        for basis in ['X', 'Y', 'Z']:
                            for CX_count in range(cnot_counts[j]):
                                #self.cnot(pair[0], pair[1])
                                self.cnot_Xtalk_only(pair[0], pair[1])
                                
                            vaccine_qubits_for_pair = []
                            for s in vaccine_qubits:
                                #if s not in pair:
                                vaccine_qubits_for_pair.append(s)
                                    
                            for q in vaccine_qubits_for_pair:
                                if basis == 'X':
                                    self.h(q)
                                elif basis == 'Y':
                                    self.sdg(q)
                                    self.h(q)
                                pauli_basis_exepected_values[pair][q][j][basis] += 2*(0.5-self.reset(q))/sample_size

        else:
            
            for pair in attack_origins:
                
                vaccine_qubits_for_pair = []
                for s in vaccine_qubits:
                    if s not in pair:
                        vaccine_qubits_for_pair.append(s)
                        
                for q in vaccine_qubits_for_pair:
                    print(q)
                    not_q = [k for k in vaccine_qubits if k!=q]
                    self.rho = self.mdl.prep_blks["layers"]["rho0"]

                    #First point (apart from north pole |0>)
                    for i in range(sample_size):
                        for j in range(2):
                            for basis in ['X', 'Y', 'Z']:
                                for CX_count in range(cnot_counts[j]):
                                    #self.cnot(pair[0], pair[1])
                                    self.cnot_Xtalk_only(pair[0], pair[1])
                                if basis == 'X':
                                    self.h(q)
                                elif basis == 'Y':
                                    self.sdg(q)
                                    self.h(q)
                                #for k in not_q:################!!!!
                                #    self.reset(k)
                                pauli_basis_exepected_values[pair][q][j][basis] += 2*(0.5-self.reset(q))/sample_size
        
        
        return pauli_basis_exepected_values
    
    @staticmethod
    def Xtalk_vaccine_parameters(pauli_basis_exepected_values, cnot_count1=2, cnot_count2=4):
        
        #find the rotation axis/centre of circle
        
        vaccine_parameters = {pair: {q: [] 
                                     for q in q_dict.keys()} 
                              for pair, q_dict in pauli_basis_exepected_values.items()}
        
        for pair, q_dict in pauli_basis_exepected_values.items():
            for q, basis_dict_list in q_dict.items():
                #three known coordinates from characterisation circuit
                coordinate_0 = np.array([0,0,1])
                coordinate_1 = np.array([basis_dict_list[0]['X'],
                                         basis_dict_list[0]['Y'],
                                         basis_dict_list[0]['Z']])
                
                coordinate_2 = np.array([basis_dict_list[1]['X'],
                                         basis_dict_list[1]['Y'],
                                         basis_dict_list[1]['Z']])
                #triangle vectors
                vec_u = coordinate_1 - coordinate_0
                vec_v = coordinate_2 - coordinate_0
                vec_w = coordinate_2 - coordinate_1
                #triangle side lengths
                u_norm = norm(vec_u)
                v_norm = norm(vec_v)
                w_norm = norm(vec_w)
                
                s = (u_norm+v_norm+w_norm)/2
                #radius
                r = u_norm*v_norm*w_norm/(4*np.sqrt(s*(s-u_norm)*(s-v_norm)*(s-w_norm)))
                
                #Barycentric coordiantes (weight for each coordinate 0,1,2)
                weight_0 = w_norm**2 * (u_norm**2 + v_norm**2 - w_norm**2)
                weight_1 = v_norm**2 * (u_norm**2 + w_norm**2 - v_norm**2)
                weight_2 = u_norm**2 * (v_norm**2 + w_norm**2 - u_norm**2)
                #cartesian coordinates of the centre
                centre = np.column_stack((coordinate_0, coordinate_1, coordinate_2)).dot(np.hstack((weight_0, weight_1, weight_2)))
                centre /= (weight_0+weight_1+weight_2)
                #normal vector to the plane, treat as the rotation axis
                rotation_axis = np.cross(coordinate_0-centre, coordinate_2-centre)
                rotation_axis /= norm(rotation_axis)
                vaccine_parameters[pair][q].append(rotation_axis)
                
                rotation_angle = np.arccos(np.dot(coordinate_0-centre, coordinate_2-centre)/(norm(coordinate_0-centre)*norm(coordinate_2-centre)))/cnot_count2
                vaccine_parameters[pair][q].append(rotation_angle)
                
        return vaccine_parameters
        
    @staticmethod
    def euler_angles_from_axis_angle(axis, angle, euler_orientation='zyz'):
        #euler takes extrinsic axes
        axis /= norm(axis)
        
        rotation = Rotation.from_rotvec(angle*axis)
        return rotation.as_euler(euler_orientation)
    
    def apply_correction(self, q, euler_angles, euler_orientation='zyz'):
        
        for i in reversed(range(3)):
            axis = euler_orientation[i]
            angle = -euler_angles[i] #reverse the rotation direction when perform correction
            
            floor_pi40_counts = int(np.floor(abs(angle/(pi/40))))
            round_pi160_counts = round((abs(angle) - pi/40*floor_pi40_counts)/(pi/160))
            
            #nearest_reps = abs(round(angle/(pi/160)))
            
            if angle > 0:
                if axis == 'z':
                    for j in range(floor_pi40_counts):
                        self.zpi40(q)
                    for j in range(round_pi160_counts):
                        self.zpi160(q)
                    
                elif axis == 'y':
                    for j in range(floor_pi40_counts):
                        self.ypi40(q)
                    for j in range(round_pi160_counts):
                        self.ypi160(q)
            else:
                if axis == 'z':
                    for j in range(floor_pi40_counts):
                        self.zminuspi40(q)
                    for j in range(round_pi160_counts):
                        self.zminuspi160(q)
                elif axis == 'y':
                    for j in range(floor_pi40_counts):
                        self.yminspi40(q)
                    for j in range(round_pi160_counts):
                        self.yminspi160(q)
            
    
    @staticmethod
    def fidelity(rho_a, rho_b):
        return np.trace(rho_a@rho_b)
        


class DensityMatrixOperations:
    
    @staticmethod
    def fidelity(rho_a, rho_b): ###Note one of the density matrix must be pure
        #rho_a_sqrt = sqrtm(rho_a)
        #trace = np.trace(sqrtm(rho_a_sqrt@rho_b@rho_a_sqrt))
        #return trace.real**2 + trace.imag**2
        return np.real(np.trace(rho_a@rho_b))
    
    @staticmethod
    def rho_from_dict(basis_normalised_counts):
        ext_basis_list = ['II', 'IX', 'IY', 'IZ',
                          'XI', 'XX', 'XY', 'XZ',
                          'YI', 'YX', 'YY', 'YZ',
                          'ZI', 'ZX', 'ZY', 'ZZ']
        
        rho = np.zeros([4, 4], dtype=complex)
        
        s_dict = {basis: 0. for basis in ext_basis_list}
        s_dict['II'] = 1.  # S for 'II' always equals 1
        
        for basis, bit_str_counts in basis_normalised_counts.items():
            s_dict[basis] = bit_str_counts['00'] - bit_str_counts['01'] - bit_str_counts['10'] + bit_str_counts['11']
            s_dict['I' + basis[1]] += (bit_str_counts['00'] - bit_str_counts['01'] + bit_str_counts['10'] - bit_str_counts['11'])/3
            s_dict[basis[0] + 'I'] += (bit_str_counts['00'] + bit_str_counts['01'] - bit_str_counts['10'] - bit_str_counts['11'])/3
        
        for basis, s in s_dict.items():
            rho += (1/4)*s*pauli_n(basis)
            
        rho = DensityMatrixOperations.find_closest_physical(rho)
        
        return rho

    @staticmethod
    def Xtalk_filter(mixed_basis_counts):
        basis_list = ['XX','XY','XZ','YX','YY','YZ','ZX','ZY','ZZ']
        unfiltered_counts = {basis: {bin(i)[2:].zfill(2): 0 
                                     for i in range(4)} 
                             for basis in basis_list}
        filtered_counts = {basis: {bin(i)[2:].zfill(2): 0 
                                     for i in range(4)} 
                             for basis in basis_list}
        
        
        for basis, bit_string_counts in mixed_basis_counts.items():
            unfiltered_sum = 0
            filtered_sum = 0
            for bit_str, detector_counts in bit_string_counts.items():
                unfiltered_counts[basis][bit_str]  = detector_counts[0] + detector_counts[1]
                unfiltered_sum += detector_counts[0] + detector_counts[1]
                
                filtered_counts[basis][bit_str] = detector_counts[0]
                filtered_sum += detector_counts[0]
            
            for bit_str in bit_string_counts.keys():
                unfiltered_counts[basis][bit_str] /= unfiltered_sum
                filtered_counts[basis][bit_str] /= filtered_sum
            
        return unfiltered_counts, filtered_counts
    
    
    @staticmethod
    def find_closest_physical(rho):
        """Algorithm to find closest physical density matrix from Smolin et al.

        Args:
            rho (numpy2d array): (unphysical) density matrix

        Returns:
            numpy2d array: physical density matrix
        """
        rho = rho/rho.trace()
        rho_physical = np.zeros(rho.shape, dtype=complex)
        # Step 1: Calculate eigenvalues and eigenvectors
        eigval, eigvec = np.linalg.eig(rho)
        # Rearranging eigenvalues from largest to smallest
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        eigval_new = np.zeros(len(eigval), dtype=complex)

        # Step 2: Let i = number of eigenvalues and set accumulator a = 0
        i = len(eigval)
        a = 0

        while (eigval[i-1] + a/i) < 0:
            a += eigval[i-1]
            i -= 1

        # Step 4: Increment eigenvalue[j] by a/i for all j <= i
        # Note since eigval_new is initialized to be all 0 so for those j>i they are already set to 0
        for j in range(i):
            eigval_new[j] = eigval[j] + a/i
            # Step 5 Construct new density matrix
            rho_physical += eigval_new[j] * \
                np.outer(eigvec[:, j], eigvec[:, j].conjugate()) #rho = Sum(lambdai*|lambdai><lambdai|)

        return rho_physical
    
    
#if __name__ == "__main__":
#    for i in range(3):
#        prep_projector_matrices_pp(i + 1)


