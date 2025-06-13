# Copyright 2025: Danny van der Haven, dannyvdhaven@gmail.com

import matplotlib.pyplot as plt
import sys
sys.path.append('../functions')
from motion_functions import *
from contact_functions import *
from contact_laws import *


# Generate velocities and motion profile
R = 1.0
motion = my_simulate_motion(
    [0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0], # initial pos, vel, ang vel
    [1.0,0.0,0.0,0.0], [1.0,0.0,0.0,0.0], # initial ori
    0, R/2, 1, np.pi, 0.05, [2*R,0.0,0.0], # normal loading, initial branch
    0, 0, 0, 0, 0, # twist
    0, 0, 0, 0, 0, # roll
    0, 0, 0, 0, 0, # shear
    [0.0,1.0,0.0], [0.0,0.0,1.0], # roll and shear axes
    6*np.pi, 6*np.pi/200 # time
)


# Simulate contact interaction
contact_params = {'k_n':100, 
                  'k_t':50, 
                  'mu':0.5, 
                  'R_i': R,
                  'R_j': R}

results = my_simulate_contact(
    motion,
    contact_params,
    Fn_linear_elastic,
    Ft_linear_Coloumb)


# Plotting motion
plt.figure(0)
plt.plot(motion['t'], motion['x_i'][:,0],'r', label='x_i')
plt.plot(motion['t'], motion['x_j'][:,0],'b', label='x_j')
plt.plot(motion['t'], motion['v_i'][:,0],'r--', label='v_i')
plt.plot(motion['t'], motion['v_j'][:,0],'b--', label='v_j')
plt.plot(results['t'], results['u_n'],'g', label='u_n')
plt.xlabel('Time')
plt.ylabel('Position or velocity')
plt.legend()
plt.show(block=False)
plt.savefig('test_plot.png')

# Plotting forces and torques
plt.figure(1)
plt.plot(results['t'], results['F_i'][:,0],'r', label='F_i')
plt.plot(results['t'], results['F_j'][:,0],'b', label='F_j')
plt.plot(results['t'], results['T_i'][:,0],'r--', label='T_i')
plt.plot(results['t'], results['T_j'][:,0],'b--', label='T_j')
plt.xlabel('Time')
plt.ylabel('Force or torque')
plt.legend()
plt.show(block=False)

dictionaryToCSV(results, open('simulatedTestPath.txt', 'w'))

import json

def make_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    else:
        return obj

def dict_to_json(_dict_, filename):
    json.dump(make_json_serializable(_dict_), open(filename, 'w'))

def json_to_dict(filename):
    def to_ndarray(obj):
        if isinstance(obj, list):
            # Recursively convert lists of lists to arrays
            if obj and isinstance(obj[0], list):
                return np.array(obj)
            # 1D arrays
            elif obj and isinstance(obj[0], (int, float)):
                return np.array(obj)
            else:
                return [to_ndarray(i) for i in obj]
        elif isinstance(obj, dict):
            return {k: to_ndarray(v) for k, v in obj.items()}
        else:
            return obj

    with open(filename, 'r') as f:
        data = json.load(f)
    return to_ndarray(data)

dict_to_json(results,'theoreticalResult.json')

demInputs = {k: results[k] for k in ['t', 'v_i', 'v_j', 'omega_i', 'omega_j']}
dict_to_json(demInputs,'demInput.json')
writeDemInput(results, 'demInput.txt')
# End of file


