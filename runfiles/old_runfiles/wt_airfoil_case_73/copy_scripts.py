copy_names = [
'c73_t21_l15_r122_k16_g1000_n376_x1_m_p',
]

import shutil
for name in copy_names: 
    shutil.copyfile('common_runner.py',name + '.py')

# mpirun -n 188 python -m mpi4py c73_t21_l15_r122_k16_g1000_n376_x1_m_p.py