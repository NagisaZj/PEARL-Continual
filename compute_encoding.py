import numpy as np

task0_encoding = np.array([ 0.0584651 ,  0.21428917, -0.09827692, -0.04488651 , 0.03023474])
task1_encoding = np.array([ 0.06830809 , 0.20561877, -0.09655312, -0.02914274, -0.01633611])
task0_encoding = np.array([-0.04206222, -0.01067364 , 0.00808698 ,-0.02391407 , 0.03573405])
task1_encoding = np.array([-0.02725895,  0.00022156 , 0.00114428 ,-0.02054096 , 0.02878061])
print(abs(task0_encoding-task1_encoding))