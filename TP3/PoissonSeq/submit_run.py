#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import os, sys, argparse, time
from subprocess import *

baseDir = os.path.dirname(os.path.realpath(__file__))
installDir = os.path.join(baseDir, 'install')

job = '''#!/bin/bash

#SBATCH --job-name=PoissoinSeq
#SBATCH --output=output.txt
#SBATCH --error=output.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --account=ams301

#SBATCH --time=00:05:00
#SBATCH --ntasks=1

## load modules
module purge
module load gcc
module load cmake

## compilation

python build.py

## execution

./install/release/PoissonSeq @ARGS@
'''

job = job.replace('@ARGS@', ' '.join(sys.argv))
jobFile = 'job.sh'

with open(jobFile, 'w') as f:
  f.write(job)

call(['sbatch', jobFile])
os.unlink(jobFile)

