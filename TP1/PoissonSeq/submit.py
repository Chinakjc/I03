#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import os, sys, subprocess
from util import *

baseDir = os.getcwd()
installDir = os.path.join(baseDir, 'install')
version = os.path.basename(baseDir)
args = standardArgs(baseDir, False, version)
args['compiler'] = 'intel'
codeSeq = os.path.join(installDir, 'PoissonSeq')

job = '''#!/bin/bash

#SBATCH --job-name=@VERSION@
#SBATCH --output=output.txt
#SBATCH --error=output.txt

@ACC@

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=@RUNTIME@

## load modules
module purge
module load intel_compiler/oneapi_2023.2.0

## execution
./install/PoissonSeq @ARGS@

'''

s = job
s = s.replace("@ARGS@", ' '.join(argList(args)))
s = s.replace("@ACC@", queue())
s = s.replace("@VERSION@", version)
s = s.replace("@RUNTIME@", args['runtime'])

with open('job.sh', 'w') as f:
  f.write(s)

subprocess.call(['sbatch', 'job.sh'])

