#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import os, sys, subprocess
from util import *

def manage():
    job = '''#!/bin/bash

#SBATCH --job-name=build_poisson
#SBATCH --output=output_build.txt
#SBATCH --error=output_build.txt

@ACC@

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=@RUNTIME@

## load modules

module purge
module load cmake/3.19.7

module load intel_compiler/19.1.3.304
module load intel_mpi/2019.9

## execution
/mnt/beegfs/softs/intel/intelpython3/bin/python ./build.py -c intel

'''

    s = job
    s = s.replace("@ACC@", queue())

    rs = 0
    rm = 30
    rh = 0
    while rm > 60:
        rh += 1
        rm -= 60
    r = "{:02d}:{:02d}:00".format(rh, rm)
    s = s.replace("@RUNTIME@", r)

    jobFile = 'job_build.sh'
    with open(jobFile, 'w') as f:
       f.write(s)

    if os.path.exists('output_build.txt'):
       os.unlink('output_build.txt')
 
    subprocess.call(['sbatch', jobFile])

if __name__ == '__main__':
    manage()
