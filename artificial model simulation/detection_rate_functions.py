import numpy as np
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan2, sqrt, exp
from numpy.linalg import norm
from scipy.stats import uniform_direction
from pygsti.circuits import Circuit
from reset import *

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

def logit_fit(x, a, b, c):
    return a/(1+np.exp((b*x+c)))

def lin_fit(x,m,c):
    return m*x+c


def false_alarm_rate_mk2(inputs):
    rotate_qubits, total_qubits, layer,crosstalk_angle, ZZ_rate, p_phase, p_damping = inputs
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(3)
    rotation_axes_averaged_list = uniform_sphere_dist.rvs(len(total_qubits), random_state=rng)
    crosstalk_axes_dict = {total_qubits[idx]: [rotation_axes_averaged_list[idx], crosstalk_angle] for idx in range(len(total_qubits))}
    
    euler_angles_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                   for q in rotate_qubits}
    euler_angles_reversed_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                                        for q in rotate_qubits}

    for j in range(len(rotate_qubits)):
        axis = rotation_axes_averaged_list[j]
        Xtalk_axis_as_thetaphi = asSpherical(axis)[1:]

        state_preparation_azimuth_axis = Xtalk_axis_as_thetaphi[1]+pi/2
        state_preparation_azimuth_axis_xyz = asCartesian([1, pi/2, state_preparation_azimuth_axis])

        state_preparation_angle = Xtalk_axis_as_thetaphi[0]


        euler_angles = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, state_preparation_angle)
        euler_angles_reversed = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, -state_preparation_angle)

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

    test = gen_idle_model(labels_with_noise=total_qubits, labels_full=total_qubits, p_phase=p_phase, p_damping=p_damping, ZZ_rate=ZZ_rate, crosstalk_axes_dict=crosstalk_axes_dict)
    
    #ghz_circ = [('Gh',rotate_qubits[0])] + [('Gcnot', rotate_qubits[idx], rotate_qubits[idx+1]) for idx in range(len(rotate_qubits)-1)]
    #c = Circuit(ghz_circ, line_labels=total_qubits, editable=True)
    test.h(rotate_qubits[0])
    for idx in range(len(rotate_qubits)-1):
        test.cnotn(rotate_qubits[idx], rotate_qubits[idx+1])
        
        
    #rotated GHZ
    for q, gate_sequence in euler_angles_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.zpi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.zpi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.zminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.zminuspi160(q)

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                            test.ypi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                            test.ypi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                            test.yminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                            test.yminuspi160(q)
                
    for i in range(layer):
        #c.append_circuit_inplace(Circuit([('Gi',1)]))
        test.idle()

    for q, gate_sequence in euler_angles_reversed_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.zpi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.zpi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.zminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.zminuspi160(q)

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                            test.ypi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                            test.ypi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                            test.yminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                            test.yminuspi160(q)

    #ghz_circ_dagger =  Circuit(ghz_circ[::-1], line_labels=total_qubits, editable=True)
    #c.append_circuit_inplace(ghz_circ_dagger)
    #c.done_editing()
    for idx in range(len(rotate_qubits)-1)[::-1]:
        test.cnotn(rotate_qubits[idx], rotate_qubits[idx+1])
    test.h(rotate_qubits[0])
    
    
    #print("mdl will simulate probabilities using a '%s' forward simulator." % str(test.mdl.sim))
    #print(test.mdl.complete_circuit(c))
    #probabilities = test.mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
    prob0 = test.prob0(rotate_qubits[0])

    #print(probabilities)
    #prob0 = 0
    #for outcome, prob in probabilities.items():
    #    if outcome[0][0] == '0':
    #        prob0 += prob

    return prob0

def false_alarm_rate(inputs):
    rotate_qubits, total_qubits, layer,crosstalk_angle, ZZ_rate, p_phase, p_damping = inputs
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(3)
    rotation_axes_averaged_list = uniform_sphere_dist.rvs(len(total_qubits), random_state=rng)
    crosstalk_axes_dict = {total_qubits[idx]: [rotation_axes_averaged_list[idx], crosstalk_angle] for idx in range(len(total_qubits))}
    
    euler_angles_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                   for q in rotate_qubits}
    euler_angles_reversed_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                                        for q in rotate_qubits}

    for j in range(len(rotate_qubits)):
        axis = rotation_axes_averaged_list[j]
        Xtalk_axis_as_thetaphi = asSpherical(axis)[1:]

        state_preparation_azimuth_axis = Xtalk_axis_as_thetaphi[1]+pi/2
        state_preparation_azimuth_axis_xyz = asCartesian([1, pi/2, state_preparation_azimuth_axis])

        state_preparation_angle = Xtalk_axis_as_thetaphi[0]


        euler_angles = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, state_preparation_angle)
        euler_angles_reversed = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, -state_preparation_angle)

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

    test = gen_idle_model(labels_with_noise=total_qubits, labels_full=total_qubits, p_phase=p_phase, p_damping=p_damping, ZZ_rate=ZZ_rate, crosstalk_axes_dict=crosstalk_axes_dict)
                
    ghz_circ = [('Gh',rotate_qubits[0])] + [('Gcnot', rotate_qubits[idx], rotate_qubits[idx+1]) for idx in range(len(rotate_qubits)-1)]
    c = Circuit(ghz_circ, line_labels=total_qubits, editable=True)
    
    #rotated GHZ
    for q, gate_sequence in euler_angles_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                
    for i in range(layer):
        c.append_circuit_inplace(Circuit([('Gi',1)]))

    for q, gate_sequence in euler_angles_reversed_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))

    ghz_circ_dagger =  Circuit(ghz_circ[::-1], line_labels=total_qubits, editable=True)
    c.append_circuit_inplace(ghz_circ_dagger)
    c.done_editing()
    
    #print("mdl will simulate probabilities using a '%s' forward simulator." % str(test.mdl.sim))
    #print(test.mdl.complete_circuit(c))
    probabilities = test.mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
    #print(probabilities)
    prob0 = 0
    for outcome, prob in probabilities.items():
        if outcome[0][0] == '0':
            prob0 += prob

    return prob0

def detection_rate_with_Xtalk_mk2(inputs):
    rotate_qubits, total_qubits, layer, max_layers, crosstalk_angle, crosstalk_angle_two_q, ZZ_rate, p_phase, p_damping = inputs
    rng = np.random.default_rng()
    crosstalk_layers = sorted(rng.choice(list(range(1,max_layers+1)), size=layer, replace=False))
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(3)
    rotation_axes_averaged_list = uniform_sphere_dist.rvs(len(total_qubits), random_state=rng)
    crosstalk_axes_dict = {total_qubits[idx]: [rotation_axes_averaged_list[idx], crosstalk_angle] for idx in range(len(total_qubits))}
    
    #rng = np.random.default_rng()
    #uniform_sphere_dist = uniform_direction(9)
    #two_q_coherent_rot_axes_list = uniform_sphere_dist.rvs(sum(range(len(total_qubits)-1)), random_state=rng)
    #crosstalk_axes_dict|{(total_qubits[idx1], total_qubits[idx2]): [uniform_sphere_dist.rvs(1, random_state=rng)[0], crosstalk_angle_two_q] for idx2 in range(len(total_qubits))  for idx1 in range(len(total_qubits)) if idx2>idx1}
    
    euler_angles_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                   for q in rotate_qubits}
    euler_angles_reversed_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                            for q in rotate_qubits}
    euler_angles_dd_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                      for q in rotate_qubits}
                                          
    for j in range(len(rotate_qubits)):
        axis = rotation_axes_averaged_list[j]
        Xtalk_axis_as_thetaphi = asSpherical(axis)[1:]

        state_preparation_azimuth_axis = Xtalk_axis_as_thetaphi[1]+pi/2
        state_preparation_azimuth_axis_xyz = asCartesian([1, pi/2, state_preparation_azimuth_axis])

        state_preparation_angle = Xtalk_axis_as_thetaphi[0]


        euler_angles = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, state_preparation_angle)
        euler_angles_reversed = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, -state_preparation_angle)

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
                
                
        
    test = gen_idle_model(labels_with_noise=total_qubits, labels_full=total_qubits, p_phase=p_phase, p_damping=p_damping, ZZ_rate=ZZ_rate, crosstalk_axes_dict=crosstalk_axes_dict)
                
    #ghz_circ = [('Gh',rotate_qubits[0])] + [('Gcnot', rotate_qubits[idx], rotate_qubits[idx+1]) for idx in range(len(rotate_qubits)-1)]
    #c = Circuit(ghz_circ, line_labels=total_qubits, editable=True)
    test.h(rotate_qubits[0])
    for idx in range(len(rotate_qubits)-1):
        test.cnotn(rotate_qubits[idx], rotate_qubits[idx+1])
        
    #rotated GHZ
    for q, gate_sequence in euler_angles_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.zpi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.zpi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.zminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.zminuspi160(q)

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                            test.ypi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                            test.ypi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                            test.yminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                            test.yminuspi160(q)
    
    
        
    for i in range(1, max_layers+1):
        if i in [1,3,5,7]:
            for q, gate_sequence in euler_angles_dd_decompositions.items():
                if q in rotate_qubits[1::2]:
                    for l in range(len(gate_sequence)):
                        if l == 0 or l == 2:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                        test.zpi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                        test.zpi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                        test.zminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                        test.zminuspi160(q)

                        else:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                        test.ypi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                        test.ypi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                        test.yminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                        test.yminuspi160(q)
    
        if i in crosstalk_layers:
            test.crosstalk()
            #c.append_circuit_inplace(Circuit([('Gcrosstalk',1)]))
        #c.append_circuit_inplace(Circuit([('Gi',1)]))
        test.idle()
        
        if i in [1,3,5,7]:
            for q, gate_sequence in euler_angles_dd_decompositions.items():
                if q in rotate_qubits[::2]:
                    for l in range(len(gate_sequence)):
                        if l == 0 or l == 2:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                        test.zpi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                        test.zpi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                        test.zminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                        test.zminuspi160(q)

                        else:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                        test.ypi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                        test.ypi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                        test.yminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                        test.yminuspi160(q)
        
        
        
    for q, gate_sequence in euler_angles_reversed_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.zpi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.zpi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.zminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.zminuspi160(q)

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                            test.ypi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                            test.ypi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                            test.yminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                            test.yminuspi160(q)

    #ghz_circ_dagger =  Circuit(ghz_circ[::-1], line_labels=total_qubits, editable=True)
    #c.append_circuit_inplace(ghz_circ_dagger)
    #c.done_editing()
    for idx in range(len(rotate_qubits)-1)[::-1]:
        test.cnotn(rotate_qubits[idx], rotate_qubits[idx+1])
    test.h(rotate_qubits[0])
    
    #print("mdl will simulate probabilities using a '%s' forward simulator." % str(test.mdl.sim))
    #print(test.mdl.complete_circuit(c))
    #probabilities = test.mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
    #print(probabilities)
    
    prob1 = test.prob1(rotate_qubits[0])
    
    #prob1 = 0
    #for outcome, prob in probabilities.items():
    #    if outcome[0][0] == '1':
    #        prob1 += prob
            
    return prob1


def detection_rate_with_Xtalk(inputs):
    rotate_qubits, total_qubits, layer, max_layers, crosstalk_angle, crosstalk_angle_two_q, ZZ_rate, p_phase, p_damping = inputs
    rng = np.random.default_rng()
    crosstalk_layers = sorted(rng.choice(list(range(1,max_layers+1)), size=layer, replace=False))
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(3)
    rotation_axes_averaged_list = uniform_sphere_dist.rvs(len(total_qubits), random_state=rng)
    crosstalk_axes_dict = {total_qubits[idx]: [rotation_axes_averaged_list[idx], crosstalk_angle] for idx in range(len(total_qubits))}
    
    #rng = np.random.default_rng()
    #uniform_sphere_dist = uniform_direction(9)
    #two_q_coherent_rot_axes_list = uniform_sphere_dist.rvs(sum(range(len(total_qubits)-1)), random_state=rng)
    #crosstalk_axes_dict|{(total_qubits[idx1], total_qubits[idx2]): [uniform_sphere_dist.rvs(1, random_state=rng)[0], crosstalk_angle_two_q] for idx2 in range(len(total_qubits))  for idx1 in range(len(total_qubits)) if idx2>idx1}
    
    euler_angles_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                   for q in rotate_qubits}
    euler_angles_reversed_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                            for q in rotate_qubits}

    for j in range(len(rotate_qubits)):
        axis = rotation_axes_averaged_list[j]
        Xtalk_axis_as_thetaphi = asSpherical(axis)[1:]

        state_preparation_azimuth_axis = Xtalk_axis_as_thetaphi[1]+pi/2
        state_preparation_azimuth_axis_xyz = asCartesian([1, pi/2, state_preparation_azimuth_axis])

        state_preparation_angle = Xtalk_axis_as_thetaphi[0]


        euler_angles = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, state_preparation_angle)
        euler_angles_reversed = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, -state_preparation_angle)

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

    test = gen_idle_model(labels_with_noise=total_qubits, labels_full=total_qubits, p_phase=p_phase, p_damping=p_damping, ZZ_rate=ZZ_rate, crosstalk_axes_dict=crosstalk_axes_dict)
                
    ghz_circ = [('Gh',rotate_qubits[0])] + [('Gcnot', rotate_qubits[idx], rotate_qubits[idx+1]) for idx in range(len(rotate_qubits)-1)]
    c = Circuit(ghz_circ, line_labels=total_qubits, editable=True)
    
    #rotated GHZ
    for q, gate_sequence in euler_angles_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                
    for i in range(1, max_layers+1):
        if i in crosstalk_layers:
            c.append_circuit_inplace(Circuit([('Gcrosstalk',1)]))
        c.append_circuit_inplace(Circuit([('Gi',1)]))

    for q, gate_sequence in euler_angles_reversed_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))

    ghz_circ_dagger =  Circuit(ghz_circ[::-1], line_labels=total_qubits, editable=True)
    c.append_circuit_inplace(ghz_circ_dagger)
    c.done_editing()
    
    #print("mdl will simulate probabilities using a '%s' forward simulator." % str(test.mdl.sim))
    #print(test.mdl.complete_circuit(c))
    probabilities = test.mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
    #print(probabilities)
    prob1 = 0
    for outcome, prob in probabilities.items():
        if outcome[0][0] == '1':
            prob1 += prob
            
    return prob1


def detection_rate_with_two_q_Xtalk_mk2(inputs):
    rotate_qubits, total_qubits, layer, max_layers, crosstalk_angle, crosstalk_angle_two_q, ZZ_rate, p_phase, p_damping = inputs
    
    rng = np.random.default_rng()
    crosstalk_layers = sorted(rng.choice(list(range(1,max_layers+1)), size=layer, replace=False))
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(3)
    rotation_axes_averaged_list = uniform_sphere_dist.rvs(len(total_qubits), random_state=rng)
    crosstalk_axes_dict = {total_qubits[idx]: [rotation_axes_averaged_list[idx], crosstalk_angle] for idx in range(len(total_qubits))}
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(9)
    #two_q_coherent_rot_axes_list = uniform_sphere_dist.rvs(sum(range(len(total_qubits)-1)), random_state=rng)
    crosstalk_axes_dict = crosstalk_axes_dict|{(total_qubits[idx1], total_qubits[idx2]): [uniform_sphere_dist.rvs(1, random_state=rng)[0], crosstalk_angle_two_q] for idx2 in range(len(total_qubits))  for idx1 in range(len(total_qubits)) if idx2>idx1}
    
    euler_angles_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                   for q in rotate_qubits}
    euler_angles_reversed_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                            for q in rotate_qubits}
    euler_angles_dd_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                      for q in rotate_qubits}
                                      
    for j in range(len(rotate_qubits)):
        axis = rotation_axes_averaged_list[j]
        Xtalk_axis_as_thetaphi = asSpherical(axis)[1:]

        state_preparation_azimuth_axis = Xtalk_axis_as_thetaphi[1]+pi/2
        state_preparation_azimuth_axis_xyz = asCartesian([1, pi/2, state_preparation_azimuth_axis])

        state_preparation_angle = Xtalk_axis_as_thetaphi[0]


        euler_angles = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, state_preparation_angle)
        euler_angles_reversed = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, -state_preparation_angle)

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
                #print(rotate_qubits)
        
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
        
        
        
    test = gen_idle_model(labels_with_noise=total_qubits, labels_full=total_qubits, p_phase=p_phase, p_damping=p_damping, ZZ_rate=ZZ_rate, crosstalk_axes_dict=crosstalk_axes_dict)
    
    #ghz_circ = [('Gh',rotate_qubits[0])] + [('Gcnot', rotate_qubits[idx], rotate_qubits[idx+1]) for idx in range(len(rotate_qubits)-1)]
    #c = Circuit(ghz_circ, line_labels=total_qubits, editable=True)
    test.h(rotate_qubits[0])
    for idx in range(len(rotate_qubits)-1):
        test.cnotn(rotate_qubits[idx], rotate_qubits[idx+1])
    #rotated GHZ
    for q, gate_sequence in euler_angles_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.zpi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.zpi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.zminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.zminuspi160(q)

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.ypi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.ypi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.yminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.yminuspi160(q)
    
    
    
    for i in range(1, max_layers+1):
        if i in [1,3,5,7]:
            for q, gate_sequence in euler_angles_dd_decompositions.items():
                if q in rotate_qubits[1::2]:
                    for l in range(len(gate_sequence)):
                        if l == 0 or l == 2:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                        test.zpi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                        test.zpi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                        test.zminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                        test.zminuspi160(q)

                        else:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                        test.ypi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                        test.ypi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                        test.yminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                        test.yminuspi160(q)
        
        if i in crosstalk_layers:
            #c.append_circuit_inplace(Circuit([('Gcrosstalk',1)]))
            test.crosstalk()
        #c.append_circuit_inplace(Circuit([('Gi',1)]))
        test.idle()
        
        if i in [1,3,5,7]:
            for q, gate_sequence in euler_angles_dd_decompositions.items():
                if q in rotate_qubits[::2]:
                    for l in range(len(gate_sequence)):
                        if l == 0 or l == 2:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                        test.zpi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                        test.zpi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                        test.zminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                                        test.zminuspi160(q)

                        else:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                        test.ypi10(q)
                                    elif rot_type == 'pi160':
                                        #c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                        test.ypi160(q)
                                    elif rot_type == '-pi10':
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                        test.yminuspi10(q)
                                    else:
                                        #c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                        test.yminuspi160(q)
                                        
        

    for q, gate_sequence in euler_angles_reversed_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.zpi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.zpi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.zminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.zminuspi160(q)

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            #c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                            test.ypi10(q)
                        elif rot_type == 'pi160':
                            #c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                            test.ypi160(q)
                        elif rot_type == '-pi10':
                            #c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                            test.yminuspi10(q)
                        else:
                            #c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))
                            test.yminuspi160(q)

    #ghz_circ_dagger =  Circuit(ghz_circ[::-1], line_labels=total_qubits, editable=True)
    #c.append_circuit_inplace(ghz_circ_dagger)
    #c.done_editing()
    for idx in range(len(rotate_qubits)-1)[::-1]:
        test.cnotn(rotate_qubits[idx], rotate_qubits[idx+1])
    test.h(rotate_qubits[0])
    
    #print("mdl will simulate probabilities using a '%s' forward simulator." % str(test.mdl.sim))
    #print(test.mdl.complete_circuit(c))
    #probabilities = test.mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
    #print(probabilities)
    
    prob1 = test.prob1(rotate_qubits[0])
    #prob1 = 0
    #for outcome, prob in probabilities.items():
    #    if outcome[0][0] == '1':
    #        prob1 += prob
    #print(f'prob1 is {prob1}')    
    return prob1


def detection_rate_with_two_q_Xtalk(inputs):
    rotate_qubits, total_qubits, layer, max_layers, crosstalk_angle, crosstalk_angle_two_q, ZZ_rate, p_phase, p_damping = inputs
    
    rng = np.random.default_rng()
    crosstalk_layers = sorted(rng.choice(list(range(1,max_layers+1)), size=layer, replace=False))
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(3)
    rotation_axes_averaged_list = uniform_sphere_dist.rvs(len(total_qubits), random_state=rng)
    crosstalk_axes_dict = {total_qubits[idx]: [rotation_axes_averaged_list[idx], crosstalk_angle] for idx in range(len(total_qubits))}
    
    rng = np.random.default_rng()
    uniform_sphere_dist = uniform_direction(9)
    #two_q_coherent_rot_axes_list = uniform_sphere_dist.rvs(sum(range(len(total_qubits)-1)), random_state=rng)
    crosstalk_axes_dict = crosstalk_axes_dict|{(total_qubits[idx1], total_qubits[idx2]): [uniform_sphere_dist.rvs(1, random_state=rng)[0], crosstalk_angle_two_q] for idx2 in range(len(total_qubits))  for idx1 in range(len(total_qubits)) if idx2>idx1}
    
    euler_angles_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                   for q in rotate_qubits}
    euler_angles_reversed_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                            for q in rotate_qubits}
    euler_angles_dd_decompositions = {q: [{'pi10':0, 'pi160':0, '-pi10':0, '-pi160':0} for i in range(3)] 
                                      for q in rotate_qubits}
                                      
    for j in range(len(rotate_qubits)):
        axis = rotation_axes_averaged_list[j]
        Xtalk_axis_as_thetaphi = asSpherical(axis)[1:]

        state_preparation_azimuth_axis = Xtalk_axis_as_thetaphi[1]+pi/2
        state_preparation_azimuth_axis_xyz = asCartesian([1, pi/2, state_preparation_azimuth_axis])

        state_preparation_angle = Xtalk_axis_as_thetaphi[0]


        euler_angles = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, state_preparation_angle)
        euler_angles_reversed = QuantumSim.euler_angles_from_axis_angle(state_preparation_azimuth_axis_xyz, -state_preparation_angle)

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
                #print(rotate_qubits)
        
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
                
                
    test = gen_idle_model(labels_with_noise=total_qubits, labels_full=total_qubits, p_phase=p_phase, p_damping=p_damping, ZZ_rate=ZZ_rate, crosstalk_axes_dict=crosstalk_axes_dict)
    
    ghz_circ = [('Gh',rotate_qubits[0])] + [('Gcnot', rotate_qubits[idx], rotate_qubits[idx+1]) for idx in range(len(rotate_qubits)-1)]
    c = Circuit(ghz_circ, line_labels=total_qubits, editable=True)
    
    #rotated GHZ
    for q, gate_sequence in euler_angles_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))


    for i in range(1, max_layers+1):
        if i in [1,3,5,7]:
            for q, gate_sequence in euler_angles_dd_decompositions.items():
                if q in rotate_qubits[1::2]:
                    for l in range(len(gate_sequence)):
                        if l == 0 or l == 2:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                    elif rot_type == 'pi160':
                                        c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                    elif rot_type == '-pi10':
                                        c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                    else:
                                        c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

                        else:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                    elif rot_type == 'pi160':
                                        c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                    elif rot_type == '-pi10':
                                        c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                    else:
                                        c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
                                    
        if i in crosstalk_layers:
            c.append_circuit_inplace(Circuit([('Gcrosstalk',1)]))
        c.append_circuit_inplace(Circuit([('Gi',1)]))
        
        if i in [1,3,5,7]:
            for q, gate_sequence in euler_angles_dd_decompositions.items():
                if q in rotate_qubits[::2]:
                    for l in range(len(gate_sequence)):
                        if l == 0 or l == 2:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                                    elif rot_type == 'pi160':
                                        c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                                    elif rot_type == '-pi10':
                                        c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                                    else:
                                        c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

                        else:
                            for rot_type, count in gate_sequence[l].items():
                                for m in range(count):
                                    if rot_type == 'pi10':
                                        c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                                    elif rot_type == 'pi160':
                                        c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                                    elif rot_type == '-pi10':
                                        c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                                    else:
                                        c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))
            

    for q, gate_sequence in euler_angles_reversed_decompositions.items():
        for l in range(len(gate_sequence)):
            if l == 0 or l == 2:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gzpi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gzpi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gz-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gz-pi160', q))], line_labels=total_qubits))

            else:
                for rot_type, count in gate_sequence[l].items():
                    for m in range(count):
                        if rot_type == 'pi10':
                            c.append_circuit_inplace(Circuit([(('Gypi10', q))], line_labels=total_qubits))
                        elif rot_type == 'pi160':
                            c.append_circuit_inplace(Circuit([(('Gypi160', q))], line_labels=total_qubits))
                        elif rot_type == '-pi10':
                            c.append_circuit_inplace(Circuit([(('Gy-pi10', q))], line_labels=total_qubits))
                        else:
                            c.append_circuit_inplace(Circuit([(('Gy-pi160', q))], line_labels=total_qubits))

    ghz_circ_dagger =  Circuit(ghz_circ[::-1], line_labels=total_qubits, editable=True)
    c.append_circuit_inplace(ghz_circ_dagger)
    c.done_editing()
    
    #print("mdl will simulate probabilities using a '%s' forward simulator." % str(test.mdl.sim))
    #print(test.mdl.complete_circuit(c))
    probabilities = test.mdl.probabilities(c) # Compute the outcome probabilities of circuit `c`
    #print(probabilities)
    prob1 = 0
    for outcome, prob in probabilities.items():
        if outcome[0][0] == '1':
            prob1 += prob
    #print(f'prob1 is {prob1}')    
    return prob1


def horizontal_mesh_centres_to_corners(mesh):
    gap = mesh[0][1] - mesh[0][0]
    row = mesh[0]
    new_mesh = []
    for i in range(len(mesh)+1):
        for column in row:
            new_mesh.append(column-gap/2)
        new_mesh.append(row[-1]+gap/2)
        
    new_mesh = np.reshape(new_mesh, (len(mesh)+1, len(row)+1))
    return new_mesh
    
def vertical_mesh_centres_to_corners(mesh):
    gap = mesh[0][0] - mesh[1][0]
    
    new_mesh = []
    for i in range(len(mesh)):
        value = mesh[i][0]
        for j in range(len(mesh[i])+1):
            new_mesh.append(value-gap/2)
    
    value = mesh[-1][0]
    for j in range(len(mesh[-1])+1):
        new_mesh.append(value+gap/2)
    
    new_mesh = np.reshape(new_mesh, (len(mesh)+1, len(mesh[-1])+1))
    return new_mesh