import os
import pickle
import numpy as np

transitions = []
folder = '../../minimuse/output_dataset/'
for filename in os.listdir(folder):
    #file_name = '00000000_00000.pkl'
    if filename != 'stats.pkl':
        with (open(folder+filename, "rb")) as openfile:
            while True:
                try:
                    step = pickle.load(openfile)
                    del step[0]['rgb_top_camera']
                    
                    transitions.append(step)
                except EOFError:
                    break

np.save('transitions',transitions)