import numpy as np
import os

# get the name of the parent directory
dir_exp = os.path.basename(os.path.dirname(os.getcwd()))


# define subjects and runs
subs_fitted = ['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']
runs = np.arange(0, 100)


# write lines
list_lines = []
for isub, sub in enumerate(subs_fitted):
    for irun, run in enumerate(runs):
        text = f"-filename=sub_{sub}_run_{run} \
-sub={sub} \
-run={run}"
        list_lines.append(text)

print("Number of simulations:", len(list_lines))

np.random.seed(21)
random_array = np.random.randint(0, 50000, size=(len(list_lines),))


# create file
with open(f'parameters.txt', 'w+') as fo:
    for ia, a in enumerate(list_lines):
        fo.write(a)
        fo.write(f' -seed={random_array[ia]}\n')
print(str(fo), 'created')