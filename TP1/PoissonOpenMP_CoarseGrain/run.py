#! /usr/bin/env python3
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str


import os, sys, subprocess
from util import *

def manage():
    baseDir = os.getcwd()
    version = os.path.basename(baseDir)
    
    args = standardArgs(baseDir, True, version)
    installDir = os.path.join(baseDir, 'install')
    
    if version == "PoissonSeq":
        codeRef = os.path.join(installDir, "PoissonSeq")
        codePar = None
    elif version.startswith("PoissonOpenMP"):
        codeRef = os.path.join(installDir, "PoissonSeq")
        codePar = os.path.join(installDir, version)
    elif version.startswith("PoissonMPI"):
        codeRef = None
        codePar = os.path.join(installDir, version)

    c = commands(codeRef, codePar, args)
    print(os.path.basename(baseDir) + ' :', file=sys.stderr)

    logFile = 'log_' + args['version'] + '.txt'
    with open(logFile, 'w') as log:
        for l in c:
            print(' '.join(l), file=sys.stderr)
            proc = subprocess.Popen(l, stdout=PIPE, stderr=STDOUT)
            while proc.poll() is None:
                text = proc.stdout.readline().decode(sys.stdout.encoding) 
                log.write(text)
                sys.stderr.write(text)

if __name__ == '__main__':
    manage()

