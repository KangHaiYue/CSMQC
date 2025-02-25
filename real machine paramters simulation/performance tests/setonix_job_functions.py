import pickle
import random
import warnings

import pygsti
from pygsti.circuits import Circuit
import numpy as np
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan2, sqrt, exp
from numpy.linalg import norm
from numpy.random import default_rng
#import matplotlib.pyplot as plt
#from matplotlib import cm, colors

#from mpl_toolkits.mplot3d import Axes3D

from reset import QuantumSim, prep_projector_matrices_pp
from reset import DensityMatrixOperations as rho_op
#from qiskit.visualization import plot_bloch_multivector, plot_bloch_vector
from math_objects import *


#warnings.filterwarnings("ignore")

def asSpherical(xyz):
    #takes list xyz (single coord)
    x       = xyz[0]
    y       = xyz[1]
    z       = xyz[2]
    r       =  sqrt(x*x + y*y + z*z)
    theta   =  arccos(z/r)
    phi     =  arctan2(y,x)
    return [r,theta,phi]

def asCartesian(rthetaphi):
    #takes list rthetaphi (single coord)
    r       = rthetaphi[0]
    theta   = rthetaphi[1]
    phi     = rthetaphi[2]
    x = r * sin( theta ) * cos( phi )
    y = r * sin( theta ) * sin( phi )
    z = r * cos( theta )
    return [x,y,z]

def distance_on_sphere(coord_a, coord_b):
    #coordinates must have norm = 1
    #coord_a /= norm(coord_a)
    #coord_b /= norm(coord_b)
    return arccos(np.dot(coord_a, coord_b))
    
def total_distance(xyz, coordiantes):
    #xyz /= norm(xyz)
    distance = 0
    for coordinate in coordiantes:
        distance += distance_on_sphere(xyz, coordinate)
    
    return distance

def con(xyz):
    return xyz[0]**2 + xyz[1]**2 + xyz[2]**2 - 1


def run_crosstalk_GHZ_4SQ_random_gates_trial(shots, basis, euler_angles, euler_angles_reversed, euler_angles_dd_decompositions, random_gate_names, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        #print(i)
        idl_tom = QuantumSim(labels=[2,3,4,5,7,8])
    
        # SQs GHZ state preparation
        idl_tom.h(2)
        idl_tom.cnot(2,3)
        idl_tom.cnot(3,5)
        idl_tom.cnot(5,8)
        # DD
        idl_tom.x(2)
        idl_tom.x(3)
        idl_tom.x(5)
        idl_tom.x(8)
        for q, gate_sequence in euler_angles.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        #Initialize DQs
        idl_tom.reset(4)
        idl_tom.reset(7)
    
        if i < shots*Xtalk_rate:
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
            
            for j in range(1,8):
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [3,8]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
                #Xtalk
                if j in crosstalk_layers and crosstalk_layers.index(j)%2 == 0:
                    idl_tom.cnot_Xtalk_only(4,8)
                elif j in crosstalk_layers and crosstalk_layers.index(j)%2 == 1:
                    idl_tom.cnot_Xtalk_only(8,4)
                # DD bottom sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [2,5]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
                                                
                                                
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
        
        else:
            for j in range(1,8):
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        for l in range(len(gate_sequence)):
                            if l == 0 or l == 2:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                            idl_tom.zpi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                            idl_tom.zpi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi160(q)
                            else:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                            idl_tom.ypi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                            idl_tom.ypi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi160(q)
                                
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
        
        
        
        #measure results in corresponding basis
        basis_1, basis_2 = basis[0], basis[1]
        if basis_1 == 'X':
            idl_tom.h(4)
        elif basis_1 == 'Y':
            idl_tom.sdg(4)
            idl_tom.h(4)

        if basis_2 == 'X':
            idl_tom.h(7)
        elif basis_2 == 'Y':
            idl_tom.sdg(7)
            idl_tom.h(7)
                
        outcome = str(idl_tom.measure(4)) + str(idl_tom.measure(7))
    
    
        #Undo GHZ
        for q, gate_sequence in euler_angles_reversed.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        idl_tom.cnot(5,8)
        idl_tom.cnot(3,5)
        idl_tom.cnot(2,3)
        idl_tom.h(2)
    
        #measure if there are odd (1,3,...) cnot attacks
        flag = idl_tom.measure(2)

        counts[outcome][flag] += 1
        
    return counts



def run_crosstalk_GHZ_2SQ_random_gates_trial(shots, basis, euler_angles, euler_angles_reversed, euler_angles_dd_decompositions, random_gate_names, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        #print(i)
        idl_tom = QuantumSim(labels=[2,3,4,5,7,8])
    
        # SQs GHZ state preparation
        idl_tom.h(2)
        idl_tom.cnot(2,3)
        # DD
        idl_tom.x(2)
        idl_tom.x(3)
        for q, gate_sequence in euler_angles.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        #Initialize DQs
        idl_tom.reset(4)
        idl_tom.reset(7)
    
        if i < shots*Xtalk_rate:
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
            for j in range(1,8):
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [2]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
                #Xtalk
                if j in crosstalk_layers and crosstalk_layers.index(j)%2 == 0:
                    idl_tom.cnot_Xtalk_only(4,8)
                elif j in crosstalk_layers and crosstalk_layers.index(j)%2 == 1:
                    idl_tom.cnot_Xtalk_only(8,4)
                # DD bottom sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [3]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
                                                
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
        
        else:
            for j in range(1,8):
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        for l in range(len(gate_sequence)):
                            if l == 0 or l == 2:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                            idl_tom.zpi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                            idl_tom.zpi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi160(q)
                            else:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                            idl_tom.ypi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                            idl_tom.ypi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi160(q)
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
        
        
        
        #measure results in corresponding basis
        basis_1, basis_2 = basis[0], basis[1]
        if basis_1 == 'X':
            idl_tom.h(4)
        elif basis_1 == 'Y':
            idl_tom.sdg(4)
            idl_tom.h(4)

        if basis_2 == 'X':
            idl_tom.h(7)
        elif basis_2 == 'Y':
            idl_tom.sdg(7)
            idl_tom.h(7)
                
        outcome = str(idl_tom.measure(4)) + str(idl_tom.measure(7))
    
    
        #Undo GHZ
        for q, gate_sequence in euler_angles_reversed.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        idl_tom.cnot(2,3)
        idl_tom.h(2)
    
        #measure if there are odd (1,3,...) cnot attacks
        flag = idl_tom.measure(2)

        counts[outcome][flag] += 1
        
    return counts


def run_crosstalk_GHZ_1SQ_random_gates_trial(shots, basis, euler_angles, euler_angles_reversed, euler_angles_dd_decompositions, random_gate_names, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        #print(i)
        idl_tom = QuantumSim(labels=[2,3,4,5,7,8])
    
        # SQs GHZ state preparation
        idl_tom.h(2)
        # DD
        idl_tom.x(2)
        for q, gate_sequence in euler_angles.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        #Initialize DQs
        idl_tom.reset(4)
        idl_tom.reset(7)
    
        if i < shots*Xtalk_rate:
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
            for j in range(1,8):
                # DD top sequence
                for q, gate_sequence in euler_angles_dd_decompositions.items():
                    for l in range(len(gate_sequence)):
                        if l == 0 or l == 2:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                        idl_tom.zpi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                        idl_tom.zpi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                        idl_tom.zminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                        idl_tom.zminuspi160(q)
                        else:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                        idl_tom.ypi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                        idl_tom.ypi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                        idl_tom.yminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                        idl_tom.yminuspi160(q)
                #Xtalk
                if j in crosstalk_layers and crosstalk_layers.index(j)%2 == 0:
                    idl_tom.cnot_Xtalk_only(4,8)
                elif j in crosstalk_layers and crosstalk_layers.index(j)%2 == 1:
                    idl_tom.cnot_Xtalk_only(8,4)
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
        
        else:
            for j in range(1,8):
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        for l in range(len(gate_sequence)):
                            if l == 0 or l == 2:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                            idl_tom.zpi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                            idl_tom.zpi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi160(q)
                            else:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                            idl_tom.ypi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                            idl_tom.ypi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi160(q)
                                            
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
        
        
        
        #measure results in corresponding basis
        basis_1, basis_2 = basis[0], basis[1]
        if basis_1 == 'X':
            idl_tom.h(4)
        elif basis_1 == 'Y':
            idl_tom.sdg(4)
            idl_tom.h(4)

        if basis_2 == 'X':
            idl_tom.h(7)
        elif basis_2 == 'Y':
            idl_tom.sdg(7)
            idl_tom.h(7)
                
        outcome = str(idl_tom.measure(4)) + str(idl_tom.measure(7))
    
    
        #Undo GHZ
        for q, gate_sequence in euler_angles_reversed.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        idl_tom.h(2)
    
        #measure if there are odd (1,3,...) cnot attacks
        flag = idl_tom.measure(2)

        counts[outcome][flag] += 1
        
    return counts


def run_crosstalk_old_random_gates_trial(shots, basis, random_gate_names, threshold=0.0001, wait_time=4, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        #print(i)
        idl_tom = QuantumSim(labels=[2,3,4,5,7,8])
        flags = []
            
        if i < shots*Xtalk_rate:
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
            for j in range(1,8):
                #Xtalk
                if j in crosstalk_layers and crosstalk_layers.index(j)%2 == 0:
                    idl_tom.cnot_Xtalk_only(4,8)
                elif j in crosstalk_layers and crosstalk_layers.index(j)%2 == 1:
                    idl_tom.cnot_Xtalk_only(8,4)
                
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
                    
                if j % wait_time == 0:
                    flags.append(idl_tom.reset(2))
                elif j == 7:
                    flags.append(idl_tom.reset(2))
                        
        
        else:
            for j in range(1,8):
                #random gates on DQ
                gate_name = random_gate_names[j-1]
                if gate_name[0] == 'CNOT':
                    idl_tom.cnotn(gate_name[1][0], gate_name[1][1])
                elif gate_name[0] == 'H':
                    idl_tom.h(gate_name[1][0])
                elif gate_name[0] == 'S':
                    idl_tom.s(gate_name[1][0])
                elif gate_name[0] == 'T':
                    idl_tom.t(gate_name[1][0])
                    
                if j % wait_time == 0:
                    flags.append(idl_tom.reset(2))
                elif j == 7:
                    flags.append(idl_tom.reset(2))
                        

        confidence = flags.count(1)/len(flags)
        if confidence < threshold:
            remove = 0
        else:
            remove = 1

        
        #measure results in corresponding basis
        basis_1, basis_2 = basis[0], basis[1]
        if basis_1 == 'X':
            idl_tom.h(4)
        elif basis_1 == 'Y':
            idl_tom.sdg(4)
            idl_tom.h(4)

        if basis_2 == 'X':
            idl_tom.h(7)
        elif basis_2 == 'Y':
            idl_tom.sdg(7)
            idl_tom.h(7)
            
        outcome = str(idl_tom.measure(4)) + str(idl_tom.measure(7))
        counts[outcome][remove] += 1
    
    return counts





def run_crosstalk_GHZ_IDT_trial_4SQ(shots, euler_angles, euler_angles_reversed, euler_angles_dd_decompositions, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        #print(i)
        idl_tom = QuantumSim(labels=[2,3,4,5,7,8])
    
        # SQs GHZ state preparation
        idl_tom.h(2)
        idl_tom.cnot(2,3)
        idl_tom.cnot(3,5)
        idl_tom.cnot(5,8)
        # DD
        idl_tom.x(2)
        idl_tom.x(3)
        idl_tom.x(5)
        idl_tom.x(8)
        for q, gate_sequence in euler_angles.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        #Initialize DQs
        idl_tom.reset(7)
        idl_tom.reset(4)
        #IDT on DQs
        idl_tom.h(7)
        idl_tom.cnotn(7,4)
        #Xtalk
        if i < shots*Xtalk_rate:
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
            
            for j in range(1,8):
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [3,8]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
            
                if j in crosstalk_layers and crosstalk_layers.index(j)%2 == 0:
                    idl_tom.cnot_Xtalk_only(4,8)
                elif j in crosstalk_layers and crosstalk_layers.index(j)%2 == 1:
                    idl_tom.cnot_Xtalk_only(8,4)
            #CX_cycle_num = CX_num // 2
            #for j in range(CX_cycle_num):
            #    idl_tom.cnot_Xtalk_only(4,8)
            #    idl_tom.cnot_Xtalk_only(8,4)
            #if CX_num > 0:
            #    idl_tom.cnot_Xtalk_only(4,8)\
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [2,5]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
            
        #IDT ends
        idl_tom.cnotn(7,4)
        idl_tom.h(7)
        #measure IDT results
        outcome = str(idl_tom.measure(7)) + str(idl_tom.measure(4))
    
    
        #Undo GHZ
        for q, gate_sequence in euler_angles_reversed.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        idl_tom.cnot(5,8)
        idl_tom.cnot(3,5)
        idl_tom.cnot(2,3)
        idl_tom.h(2)
    
        #measure if there are odd (1,3,...) cnot attacks
        flag = idl_tom.measure(2)

        counts[outcome][flag] += 1
        
    return counts




def run_crosstalk_GHZ_IDT_trial_2SQ(shots, euler_angles, euler_angles_reversed, euler_angles_dd_decompositions, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        #print(i)
        idl_tom = QuantumSim(labels=[2,3,4,5,7,8])
    
        # SQs GHZ state preparation
        idl_tom.h(2)
        idl_tom.cnot(2,3)
        # DD
        idl_tom.x(2)
        idl_tom.x(3)
        for q, gate_sequence in euler_angles.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        #Initialize DQs
        idl_tom.reset(7)
        idl_tom.reset(4)
        #IDT on DQs
        idl_tom.h(7)
        idl_tom.cnotn(7,4)
        #Xtalk
        if i < shots*Xtalk_rate:
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
            
            for j in range(1,8):
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [3]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
            
                if j in crosstalk_layers and crosstalk_layers.index(j)%2 == 0:
                    idl_tom.cnot_Xtalk_only(4,8)
                elif j in crosstalk_layers and crosstalk_layers.index(j)%2 == 1:
                    idl_tom.cnot_Xtalk_only(8,4)
            #CX_cycle_num = CX_num // 2
            #for j in range(CX_cycle_num):
            #    idl_tom.cnot_Xtalk_only(4,8)
            #    idl_tom.cnot_Xtalk_only(8,4)
            #if CX_num > 0:
            #    idl_tom.cnot_Xtalk_only(4,8)\
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        if q in [2]:
                            for l in range(len(gate_sequence)):
                                if l == 0 or l == 2:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                                idl_tom.zpi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                                idl_tom.zpi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                                idl_tom.zminuspi160(q)

                                else:
                                    for rot_type, count in gate_sequence[l].items():
                                        for m in range(count):
                                            if rot_type == 'pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                                idl_tom.ypi10(q)
                                            elif rot_type == 'pi160':
                                                #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                                idl_tom.ypi160(q)
                                            elif rot_type == '-pi10':
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi10(q)
                                            else:
                                                #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                                idl_tom.yminuspi160(q)
            
        #IDT ends
        idl_tom.cnotn(7,4)
        idl_tom.h(7)
        #measure IDT results
        outcome = str(idl_tom.measure(7)) + str(idl_tom.measure(4))
    
    
        #Undo GHZ
        for q, gate_sequence in euler_angles_reversed.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        idl_tom.cnot(2,3)
        idl_tom.h(2)
    
        #measure if there are odd (1,3,...) cnot attacks
        flag = idl_tom.measure(2)

        counts[outcome][flag] += 1
        
    return counts


def run_crosstalk_GHZ_IDT_trial_1SQ(shots, euler_angles, euler_angles_reversed, euler_angles_dd_decompositions, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        #print(i)
        idl_tom = QuantumSim(labels=[2,3,4,5,7,8])
    
        # SQs GHZ state preparation
        idl_tom.h(2)
        # DD
        idl_tom.x(2)
        for q, gate_sequence in euler_angles.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        #Initialize DQs
        idl_tom.reset(7)
        idl_tom.reset(4)
        #IDT on DQs
        idl_tom.h(7)
        idl_tom.cnotn(7,4)
        #Xtalk
        if i < shots*Xtalk_rate:
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
            
            for j in range(1,8):
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        for l in range(len(gate_sequence)):
                            if l == 0 or l == 2:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                            idl_tom.zpi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                            idl_tom.zpi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi160(q)
                            else:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                            idl_tom.ypi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                            idl_tom.ypi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi160(q)
            
                if j in crosstalk_layers and crosstalk_layers.index(j)%2 == 0:
                    idl_tom.cnot_Xtalk_only(4,8)
                elif j in crosstalk_layers and crosstalk_layers.index(j)%2 == 1:
                    idl_tom.cnot_Xtalk_only(8,4)
            #CX_cycle_num = CX_num // 2
            #for j in range(CX_cycle_num):
            #    idl_tom.cnot_Xtalk_only(4,8)
            #    idl_tom.cnot_Xtalk_only(8,4)
            #if CX_num > 0:
            #    idl_tom.cnot_Xtalk_only(4,8)\
                # DD top sequence
                if j in [1,3,5,7]:
                    for q, gate_sequence in euler_angles_dd_decompositions.items():
                        for l in range(len(gate_sequence)):
                            if l == 0 or l == 2:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                            idl_tom.zpi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                            idl_tom.zpi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                            idl_tom.zminuspi160(q)
                            else:
                                for rot_type, count in gate_sequence[l].items():
                                    for m in range(count):
                                        if rot_type == 'pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                            idl_tom.ypi10(q)
                                        elif rot_type == 'pi160':
                                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                            idl_tom.ypi160(q)
                                        elif rot_type == '-pi10':
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi10(q)
                                        else:
                                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                            idl_tom.yminuspi160(q)
            
        #IDT ends
        idl_tom.cnotn(7,4)
        idl_tom.h(7)
        #measure IDT results
        outcome = str(idl_tom.measure(7)) + str(idl_tom.measure(4))
    
    
        #Undo GHZ
        for q, gate_sequence in euler_angles_reversed.items():
            for j in range(len(gate_sequence)):
                if j == 0 or j == 2:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.zpi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.zpi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.zminuspi10(q)
                            else:
                                idl_tom.zminuspi160(q)

                else:
                    for rot_type, reps in gate_sequence[j].items():
                        for k in range(reps):
                            if rot_type == 'pi10':
                                idl_tom.ypi10(q)
                            elif rot_type == 'pi160':
                                idl_tom.ypi160(q)
                            elif rot_type == '-pi10':
                                idl_tom.yminuspi10(q)
                            else:
                                idl_tom.yminuspi160(q)
    
        idl_tom.h(2)
    
        #measure if there are odd (1,3,...) cnot attacks
        flag = idl_tom.measure(2)

        counts[outcome][flag] += 1
        
    return counts


def run_crosstalk_old_IDT_trial(shots, threshold=0.0001, wait_time=4, CX_num=1, Xtalk_rate=0.5):
    
    counts = {bin(i)[2:].zfill(2): {0: 0, 1: 0} 
              for i in range(4)} 
    
    for i in range(shots):
        print(i)
        outcome = []
        idl_tom_sq = QuantumSim(labels=[2,3,4,5,7,8])
        idl_tom_sq.h(7)
        idl_tom_sq.cnotn(7,4)
    
        if i < shots*Xtalk_rate:
            crosstalk = True
            rng = default_rng()
            crosstalk_layers = sorted(rng.choice(list(range(1,8)), size=CX_num, replace=False))
        else:
            crosstalk = False
        for layer in range(1, 9):
            if crosstalk:
                if layer in crosstalk_layers and crosstalk_layers.index(layer)%2 == 0:
                    idl_tom_sq.cnot_Xtalk_only(4,8)
                elif layer in crosstalk_layers and crosstalk_layers.index(layer)%2 == 1:
                    idl_tom_sq.cnot_Xtalk_only(8,4)
            if layer % wait_time == 0:
                outcome.append(idl_tom_sq.reset(2))

            
        confidence = outcome.count(1)/len(outcome)
        if confidence < threshold:
            remove = 0
        else:
            remove = 1
        idl_tom_sq.cnotn(7,4)
        idl_tom_sq.h(7)
        data_outcome = str(idl_tom_sq.measure(7)) + str(idl_tom_sq.measure(4))
    
        counts[data_outcome][remove] += 1
    
    return counts