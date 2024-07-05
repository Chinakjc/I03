#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import os, sys, subprocess
from util import *

def manage():
  baseDir = os.getcwd()
  installDir = os.path.join(baseDir, 'install')
  version = os.path.basename(baseDir)
  args = standardArgs(baseDir, False, version)
  args['compiler'] = 'intel'
  codeSeq = os.path.join(installDir, 'PoissonSeq')
  codePar = os.path.join(installDir, version)

  job = '''#!/bin/bash

#SBATCH --job-name=@VERSION@
#SBATCH --output=output.txt
#SBATCH --error=output.txt

@ACC@

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=@THREADS_MAX@
#SBATCH --time=@RUNTIME@

## load modules
module purge
module load cmake/3.19.7

module load intel_compiler/19.1.3.304
module load intel_mpi/2019.9

## execution
/mnt/beegfs/softs/intel/intelpython3/bin/python -u ./run.py -c intel @ARGS@

'''

  s = job
  s = s.replace("@ARGS@", ' '.join(argList(args)))
  s = s.replace("@ACC@", queue())
  s = s.replace("@VERSION@", version)
  s = s.replace("@THREADS_MAX@", str(args['threadsMax']))
  s = s.replace("@RUNTIME@", args['runtime'])

  with open('job.sh', 'w') as f:
    f.write(s)

  call(['sbatch', 'job.sh'])

if __name__ == '__main__':
    manage()
