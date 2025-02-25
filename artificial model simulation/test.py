#from pygsti.models.modelnoise import DepolarizationNoise
import itertools
import os
import numpy as np
from numpy import pi, sin, cos, tan, arcsin, arccos, arctan2, sqrt, exp
from numpy.linalg import norm
import scipy.optimize as opt
from scipy.stats import uniform_direction
from multiprocessing import Pool #, set_start_method, get_context
#set_start_method("spawn")
from reset import *
from detection_rate_functions import *




def false_alarm_task():
    layers_range = [1]
    #num_q_range = [2,3,4,5]
    num_q_range = [1,2,3,4,5,6]
    ZZ_rates = np.logspace(-5,0,100,base=10)
    #ZZ_rates = np.linspace(0,1,100)

    #p_phase=1.48e-4 #(T2=115us on ibm_torino) (CZ gate time = 68ns)
    p_phase=6.16e-4 #(T2=186.05us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    #p_phase=0
    #p_damping=2.17e-4 #(T1=157us on ibm_torino) (CZ gate time = 68ns)
    p_damping=1.001e-3 #(T1=266us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    #p_damping=0
    reps = 50
    max_threads = len(os.sched_getaffinity(os.getpid()))-2

    detection_rate_dict = {num_q: {layer: {rate: [] 
                                           for rate in ZZ_rates} 
                                   for layer in layers_range} 
                           for num_q in num_q_range}

    for num_q in num_q_range:
        rotate_qubits = list(range(1,num_q+1))
        print(rotate_qubits)
        total_qubits = [1,2,3,4,5,6]
        crosstalk_angle = pi/len(rotate_qubits)
        for layer in layers_range:
            for ZZ_rate in ZZ_rates:
                dummy_inputs = [list(range(reps))[x:x+max_threads] for x in range(0, reps, max_threads)]
                for k in range(len(dummy_inputs)):
                    inputs = [(rotate_qubits, 
                                total_qubits, 
                                layer,
                                crosstalk_angle, 
                                ZZ_rate, 
                                p_phase, 
                                p_damping) for x in dummy_inputs[k]]
                    with Pool(max_threads) as p:
                        for result in p.map(false_alarm_rate_mk2, inputs):
                            detection_rate_dict[num_q][layer][ZZ_rate].append(result)

                    #detection_rate_dict[num_q][layer][ZZ_rate].append(prob0)
                print(np.mean(detection_rate_dict[num_q][layer][ZZ_rate]))
    
    
    false_alarm_rate_1 = 1-np.mean([detection_rate_dict[1][1][rate] for rate in ZZ_rates], axis=1)
    false_alarm_rate_2 = 1-np.mean([detection_rate_dict[2][1][rate] for rate in ZZ_rates], axis=1)
    false_alarm_rate_3 = 1-np.mean([detection_rate_dict[3][1][rate] for rate in ZZ_rates], axis=1)
    false_alarm_rate_4 = 1-np.mean([detection_rate_dict[4][1][rate] for rate in ZZ_rates], axis=1)
    false_alarm_rate_5 = 1-np.mean([detection_rate_dict[5][1][rate] for rate in ZZ_rates], axis=1)
    false_alarm_rate_6 = 1-np.mean([detection_rate_dict[6][1][rate] for rate in ZZ_rates], axis=1)

    false_alarm_rate_1_err = np.std([detection_rate_dict[1][1][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    false_alarm_rate_2_err = np.std([detection_rate_dict[2][1][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    false_alarm_rate_3_err = np.std([detection_rate_dict[3][1][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    false_alarm_rate_4_err = np.std([detection_rate_dict[4][1][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    false_alarm_rate_5_err = np.std([detection_rate_dict[5][1][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    false_alarm_rate_6_err = np.std([detection_rate_dict[6][1][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    
    print(list(false_alarm_rate_1))
    print(list(false_alarm_rate_2))
    print(list(false_alarm_rate_3))
    print(list(false_alarm_rate_4))
    print(list(false_alarm_rate_5))
    print(list(false_alarm_rate_6))

    print(list(false_alarm_rate_1_err))
    print(list(false_alarm_rate_2_err))
    print(list(false_alarm_rate_3_err))
    print(list(false_alarm_rate_4_err))
    print(list(false_alarm_rate_5_err))
    print(list(false_alarm_rate_6_err))



def task_detection_failing_rate():
    layers_range = [1,2,3,4,5,6,7]
    max_layers = max(layers_range)
    num_q_range = [1,2,4]
    crosstalk_angle = pi/4
    crosstalk_angle_two_q = 0
    ZZ_rates = np.logspace(-5,0,100,base=10)
    #ZZ_rates = np.linspace(0,1,100)

    #p_phase=1.48e-4 #(T2=115us on ibm_torino) (CZ gate time = 68ns)
    #p_phase=6.16e-4 #(T2=186.05us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    p_phase=0
    #p_damping=2.17e-4 #(T1=157us on ibm_torino) (CZ gate time = 68ns)
    #p_damping=1.001e-3 #(T1=266us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    p_damping=0
    reps = 50
    max_threads = len(os.sched_getaffinity(os.getpid()))-2

    detection_rate_dict = {num_q: {layer: {rate: [] 
                                           for rate in ZZ_rates} 
                                   for layer in layers_range if (layer*num_q*pi/4 + pi)%(2*pi)==0} 
                           for num_q in num_q_range}



    for num_q in detection_rate_dict.keys():
        rotate_qubits = list(range(1,num_q+1))
        total_qubits = [1,2,3,4,5,6]
        #print(rotate_qubits)
        for layer in detection_rate_dict[num_q].keys():
            print(layer)
            for ZZ_rate in ZZ_rates:
                dummy_inputs = [list(range(reps))[x:x+max_threads] for x in range(0, reps, max_threads)]
                for k in range(len(dummy_inputs)):
                    inputs = [(rotate_qubits, 
                                total_qubits, 
                                layer,
                                max_layers, 
                                crosstalk_angle, 
                                crosstalk_angle_two_q, 
                                ZZ_rate, 
                                p_phase, 
                                p_damping) for x in dummy_inputs[k]]
                    with Pool(max_threads) as p:
                        for result in p.map(detection_rate_with_Xtalk_mk2, inputs):
                            detection_rate_dict[num_q][layer][ZZ_rate].append(result)

                print(np.mean(detection_rate_dict[num_q][layer][ZZ_rate]))
                
    
    detection_rates_dict = {}
    detection_rates_err_dict = {}
    detection_rates_dict[1] = np.mean([detection_rate_dict[4][1][rate] for rate in ZZ_rates], axis=1)
    detection_rates_dict[2] = np.mean([detection_rate_dict[2][2][rate] for rate in ZZ_rates], axis=1)
    detection_rates_dict[3] = np.mean([detection_rate_dict[4][3][rate] for rate in ZZ_rates], axis=1)
    detection_rates_dict[4] = np.mean([detection_rate_dict[1][4][rate] for rate in ZZ_rates], axis=1)
    detection_rates_dict[5] = np.mean([detection_rate_dict[4][5][rate] for rate in ZZ_rates], axis=1)
    detection_rates_dict[6] = np.mean([detection_rate_dict[2][6][rate] for rate in ZZ_rates], axis=1)
    detection_rates_dict[7] = np.mean([detection_rate_dict[4][7][rate] for rate in ZZ_rates], axis=1)


    detection_rates_err_dict[1] = np.std([detection_rate_dict[4][1][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    detection_rates_err_dict[2] = np.std([detection_rate_dict[2][2][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    detection_rates_err_dict[3] = np.std([detection_rate_dict[4][3][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    detection_rates_err_dict[4] = np.std([detection_rate_dict[1][4][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    detection_rates_err_dict[5] = np.std([detection_rate_dict[4][5][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    detection_rates_err_dict[6] = np.std([detection_rate_dict[2][6][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    detection_rates_err_dict[7] = np.std([detection_rate_dict[4][7][rate] for rate in ZZ_rates], axis=1)/np.sqrt(reps)
    
    print(list(detection_rates_dict[1]))
    print(list(detection_rates_dict[2]))
    print(list(detection_rates_dict[3]))
    print(list(detection_rates_dict[4]))
    print(list(detection_rates_dict[5]))
    print(list(detection_rates_dict[6]))
    print(list(detection_rates_dict[7]))


    print(list(detection_rates_err_dict[1]))
    print(list(detection_rates_err_dict[2]))
    print(list(detection_rates_err_dict[3]))
    print(list(detection_rates_err_dict[4]))
    print(list(detection_rates_err_dict[5]))
    print(list(detection_rates_err_dict[6]))
    print(list(detection_rates_err_dict[7]))
                

def task_detection_failing_rate_2d_4sq():
    layers_range = [1,2,3,4,5,6,7]
    max_layers = max(layers_range)
    num_q_range = [1,2,4]
    #crosstalk_angle = pi/4
    crosstalk_angle = 2*pi/9
    ZZ_rates = np.logspace(-3,0,20,base=10)
    #ZZ_rates = np.linspace(0,1,100)
    #ZZ_rate = 1e-3
    two_q_crosstalk_angles = np.logspace(-3,0,20,base=10)
    
    #p_phase=1.48e-4 #(T2=115us on ibm_torino) (CZ gate time = 68ns)
    #p_phase=6.16e-4 #(T2=186.05us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    p_phase=0
    #p_damping=2.17e-4 #(T1=157us on ibm_torino) (CZ gate time = 68ns)
    #p_damping=1.001e-3 #(T1=266us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    p_damping=0
    reps = 50
    max_threads = len(os.sched_getaffinity(os.getpid()))-2
    #max_threads = 5
    detection_rate_dict = {num_q: {layer: {(ZZ_rate, two_q_angle): [] 
                                           for two_q_angle in two_q_crosstalk_angles for ZZ_rate in ZZ_rates} 
                                   for layer in layers_range if (layer*num_q*pi/4 + pi)%(2*pi)==0} 
                           for num_q in num_q_range}
    #print(detection_rate_dict)
    
    for num_q in detection_rate_dict.keys():
        rotate_qubits = list(range(1,num_q+1))
        total_qubits = [1,2,3,4,5,6]
        #print(rotate_qubits)
        for layer in detection_rate_dict[num_q].keys():
            print(layer)
            for ZZ_rate in ZZ_rates:
                for crosstalk_angle_two_q in two_q_crosstalk_angles:
                    dummy_inputs = [list(range(reps))[x:x+max_threads] for x in range(0, reps, max_threads)]
                    for k in range(len(dummy_inputs)):
                        inputs = [(rotate_qubits, 
                                    total_qubits, 
                                    layer,
                                    max_layers, 
                                    crosstalk_angle, 
                                    crosstalk_angle_two_q, 
                                    ZZ_rate, 
                                    p_phase, 
                                    p_damping) for x in dummy_inputs[k]]
                        with Pool(max_threads) as p:
                            for result in p.map(detection_rate_with_two_q_Xtalk_mk2, inputs):
                                detection_rate_dict[num_q][layer][(ZZ_rate, crosstalk_angle_two_q)].append(result)
    
                    print(np.mean(detection_rate_dict[num_q][layer][(ZZ_rate, crosstalk_angle_two_q)]))
    
    detection_rates_dict = {}
    detection_rates_err_dict = {}
    detection_rates_dict[1] = np.reshape(np.mean([detection_rate_dict[4][1][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[2] = np.reshape(np.mean([detection_rate_dict[2][2][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[3] = np.reshape(np.mean([detection_rate_dict[4][3][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[4] = np.reshape(np.mean([detection_rate_dict[1][4][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[5] = np.reshape(np.mean([detection_rate_dict[4][5][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[6] = np.reshape(np.mean([detection_rate_dict[2][6][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[7] = np.reshape(np.mean([detection_rate_dict[4][7][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    
    
    detection_rates_err_dict[1] = np.reshape(np.std([detection_rate_dict[4][1][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[2] = np.reshape(np.std([detection_rate_dict[2][2][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[3] = np.reshape(np.std([detection_rate_dict[4][3][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[4] = np.reshape(np.std([detection_rate_dict[1][4][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[5] = np.reshape(np.std([detection_rate_dict[4][5][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[6] = np.reshape(np.std([detection_rate_dict[2][6][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[7] = np.reshape(np.std([detection_rate_dict[4][7][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    
    print(detection_rates_dict)
    print(detection_rates_err_dict)
    
    #Save the data
    with open(f"./detection_rates_dd/pure_detection_rates_2pi9.pkl", "wb") as f:
            pickle.dump(detection_rates_dict, f)
    with open(f"./detection_rates_dd/pure_detection_rates_2pi9_err.pkl", "wb") as f:
            pickle.dump(detection_rates_err_dict, f)
            
            
            
def task_detection_failing_rate_2d_8sq():
    layers_range = [1,2,3,4,5,6,7]
    max_layers = max(layers_range)
    #num_q_range = [1,2,4]
    num_q_range = [2,4,8]
    #crosstalk_angle = pi/4
    #crosstalk_angle = 2*pi/9
    #crosstalk_angle = pi/8
    crosstalk_angle = 2*pi/17
    ZZ_rates = np.logspace(-3,0,20,base=10)/2
    #ZZ_rates = np.linspace(0,1,100)
    #ZZ_rate = 1e-3
    two_q_crosstalk_angles = np.logspace(-3,0,20,base=10)/2
    
    #p_phase=1.48e-4 #(T2=115us on ibm_torino) (CZ gate time = 68ns)
    #p_phase=6.16e-4 #(T2=186.05us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    p_phase=0
    #p_damping=2.17e-4 #(T1=157us on ibm_torino) (CZ gate time = 68ns)
    #p_damping=1.001e-3 #(T1=266us on ibm_sherbrooke) (ECR gate time = 533.33ns)
    p_damping=0
    reps = 50
    max_threads = len(os.sched_getaffinity(os.getpid()))-2
    #max_threads = 5
    detection_rate_dict = {num_q: {layer: {(ZZ_rate, two_q_angle): [] 
                                           for two_q_angle in two_q_crosstalk_angles for ZZ_rate in ZZ_rates} 
                                   for layer in layers_range if (layer*num_q*pi/8 + pi)%(2*pi)==0} 
                           for num_q in num_q_range}
    #print(detection_rate_dict)
    
    for num_q in detection_rate_dict.keys():
        rotate_qubits = list(range(1,num_q+1))
        total_qubits = [1,2,3,4,5,6,7,8]
        #print(rotate_qubits)
        for layer in detection_rate_dict[num_q].keys():
            print(layer)
            for ZZ_rate in ZZ_rates:
                for crosstalk_angle_two_q in two_q_crosstalk_angles:
                    dummy_inputs = [list(range(reps))[x:x+max_threads] for x in range(0, reps, max_threads)]
                    for k in range(len(dummy_inputs)):
                        inputs = [(rotate_qubits, 
                                    total_qubits, 
                                    layer,
                                    max_layers, 
                                    crosstalk_angle, 
                                    crosstalk_angle_two_q, 
                                    ZZ_rate, 
                                    p_phase, 
                                    p_damping) for x in dummy_inputs[k]]
                        with Pool(max_threads) as p:
                            for result in p.map(detection_rate_with_two_q_Xtalk, inputs):
                                detection_rate_dict[num_q][layer][(ZZ_rate, crosstalk_angle_two_q)].append(result)
    
                    print(np.mean(detection_rate_dict[num_q][layer][(ZZ_rate, crosstalk_angle_two_q)]))
    
    detection_rates_dict = {}
    detection_rates_err_dict = {}
    detection_rates_dict[1] = np.reshape(np.mean([detection_rate_dict[8][1][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[2] = np.reshape(np.mean([detection_rate_dict[4][2][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[3] = np.reshape(np.mean([detection_rate_dict[8][3][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[4] = np.reshape(np.mean([detection_rate_dict[2][4][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[5] = np.reshape(np.mean([detection_rate_dict[8][5][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[6] = np.reshape(np.mean([detection_rate_dict[4][6][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_dict[7] = np.reshape(np.mean([detection_rate_dict[8][7][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    
    
    detection_rates_err_dict[1] = np.reshape(np.std([detection_rate_dict[8][1][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[2] = np.reshape(np.std([detection_rate_dict[4][2][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[3] = np.reshape(np.std([detection_rate_dict[8][3][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[4] = np.reshape(np.std([detection_rate_dict[2][4][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[5] = np.reshape(np.std([detection_rate_dict[8][5][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[6] = np.reshape(np.std([detection_rate_dict[4][6][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    detection_rates_err_dict[7] = np.reshape(np.std([detection_rate_dict[8][7][(rate, two_q_angle)] for two_q_angle in two_q_crosstalk_angles for rate in ZZ_rates], axis=1)/np.sqrt(reps), (len(two_q_crosstalk_angles), len(ZZ_rates)))
    
    print(detection_rates_dict)
    print(detection_rates_err_dict)
    
    #Save the data
    with open(f"./detection_rates_dd/pure_detection_rates_2pi17.pkl", "wb") as f:
            pickle.dump(detection_rates_dict, f)
    with open(f"./detection_rates_dd/pure_detection_rates_2pi17_err.pkl", "wb") as f:
            pickle.dump(detection_rates_err_dict, f)
            
            
if __name__ == '__main__':
    #task()
    #task_detection_failing_rate()
    #task_detection_failing_rate_2d_4sq()
    task_detection_failing_rate_2d_8sq()
    #false_alarm_task()