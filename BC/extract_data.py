import os
import os.path
import pickle
import numpy as np

n_episodes = 10
folder = '../../minimuse/output_dataset/'
for i in range(n_episodes):
    k=0
    traj = []
    while True :
        if i <10:
            filename = '0000000' + str(i) 
        elif i<100:
            filename = '000000' + str(i) 
        else:
            filename = '00000' + str(i) 

        if k<10:
            filename += '_0000' + str(k) + '.pkl'
        elif k<100:
            filename += '_000' + str(k) + '.pkl'
        else:
            filename += '_00' + str(k) + '.pkl'
        path = folder + filename
        if os.path.isfile(path):
            with (open(path, "rb")) as openfile:
                while True:
                    try:
                        step = pickle.load(openfile)
                        del step[0]['rgb_top_camera']
                        
                        traj.append(step)
                    except EOFError:
                        break
            k+=1
        else:
            print(path)
            np.save('trajs/traj'+str(i),traj)
            break

