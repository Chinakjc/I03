#! /usr/bin/env python
# -*- coding: utf-8 -*-

try:
    unicode('')
except NameError:
    unicode = str

import time, argparse, sys, os, copy, platform
from subprocess import *

acc_in = '''
#SBATCH --partition=cpu_dist
#SBATCH --reservation=ams301_csp
#SBATCH --account=ams301
'''

acc_out = '''
#SBATCH --partition=cpu_dist
#SBATCH --partition=cpu_test
##SBATCH --reservation=ams301_csp
#SBATCH --account=ams301
'''

# annee, mois, jour, debut(heure.minute), fin(heure.minute)
courses = [
  [2022, 11, 25,  9.0, 12.16],
  [2022, 12,  2,  9.0, 12.16],
  [2022, 12,  9,  9.0, 12.16],
  [2022, 12, 16,  9.0, 12.16],
  [2023,  1,  6,  9.0, 12.16],
  [2023,  1, 13,  9.0, 12.16],
  [2023,  1, 20,  9.0, 12.16],
  [2023,  1, 27,  9.0, 12.16],
  [2023,  2,  3,  9.0, 12.16],
  [2023,  2, 10,  9.0, 12.16]
]


def queue():
  t = time.localtime()
  hm = t.tm_hour + 0.01 * t.tm_min
  
  inCourse = False
  for tc in courses:
    if t.tm_year == tc[0] and t.tm_mon == tc[1] and t.tm_mday == tc[2]:
      
      if hm > tc[3] and hm < tc[4]:
        inCourse = True
        break

  if inCourse:
     return acc_in
  else:
     return acc_out

class Range(object):
    def __init__(self, start, end = None):
        self.start = start
        self.end = end

    def __eq__(self, other):
        if self.end:
            return self.start <= other <= self.end
        else:
            return self.start <= other

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield self

    def __str__(self):
        if self.end:
            return 'in ]{0},{1}]'.format(self.start, self.end)
        else:
            return '> {0}'.format(self.start)

def standardArgs(baseDir, direct, version):
    parser = argparse.ArgumentParser()
    p = platform.system()
    if p == 'Windows':
      defaultCompiler = 'intel'
    else:
      defaultCompiler = 'gnu'
    
    parser.add_argument('--compiler', '-c', choices=['gnu', 'intel'], default=defaultCompiler)
    parser.add_argument('--n', type=int, choices=Range(2), default=401, help="number of points in each direction (default 401)")
    parser.add_argument('--n0', type=int, choices=Range(2), default=401, help="number of points in X direction (default 401)")
    parser.add_argument('--n1', type=int, choices=Range(2), default=401, help="number of points in Y direction (default 401)")
    parser.add_argument('--n2', type=int, choices=Range(2), default=401, help="number of points in Z direction (default 401)")
    parser.add_argument('--dt', type=float, choices=Range(0.0), help='time step (positive decimal number, dafault computed value)')
    parser.add_argument('--it', type=int, choices=Range(0), default=10, help='number of time steps (default 10)')
    parser.add_argument('--out', type=int, default=0, help='output frequency (default 0 - no output)')
    parser.add_argument('--graphics', type=bool, default=False)
    if not direct:
        parser.add_argument('--runtime', '-r', type=int, default=10, help='maximum running time (default 10 minutes)')
    
    if version.startswith("PoissonOpenMP"):
        parser.add_argument('--threadsMax', type=int, default=8, help='number of threads')
    if "Coarse" in version:
        parser.add_argument('--balances', type=int, default=0, help='number of load balancing steps')
    if "MPI" in version:
        parser.add_argument('--npMax', type=int, default=8, help='number of processes')

    args = parser.parse_args()
    args.baseDir = baseDir
    
    args = checkArgs(args, direct, version)
    return args

def checkArgs(args, direct, version):
    args2 = {}
    for a in dir(args):
        if not (a[0] == "_" or a == 'baseDir'):
            v = eval('args.' + a)
            if not v is None:
                args2[a] = v

    if not direct:
        rs = 0
        rm = args.runtime 
        rh = 0
        while rm > 60:
          rh += 1
          rm -= 60
        args2['runtime'] = "{:02d}:{:02d}:00".format(rh, rm)

    if not (args2['n0'] == 401 and args2['n1'] == 401 and args2['n2'] == 401):
        del args2['n']
    else:
        del args2['n0']
        del args2['n1']
        del args2['n2']

    args2['version'] = version
    if not 'OpenMP' in version:
        args2['threadsMax'] = 1

    if 'MPI' in version and 'OpenMP' in version:
        pass

    return args2

def readTemps(n=None, t=None): 
    s = os.path.join('results', 'temps')
    if n:
       s = s + '_n_' +str(n)
    if t:
        s += '_t_' + str(t)
    s += '.dat'
    v = (0.0, 0.0)
    with open(s) as f:
        l = f.readline().split()
        v = (float(l[2]), float(l[3]))
    return v

def argList(args):
    l = []
    for a in args:
        if not a in ['version', 'runtime', 'threadsMax', 'npMax']:
            l += ["--" + a, str(args[a])]
    return l


def command(code, args, prefix = []):
    return prefix  + [code] + argList(args)


def commands(codeRef, codePar, args):

    c = []
    prefix = []

    if 'MPI' in args['version']:
      nmax = args['npMax']
      prefix = ['mpiexec']
    else:
      nmax = 1

    args2 = copy.deepcopy(args)
    for a in ['version', 'graphics', 'compiler', 'threadsMax', 'npMax']:
        if a in args2:
            del args2[a]
           
    if codeRef:
        c.append(command(codeRef, args2, prefix))

    if args['version'].startswith("PoissonOpenMP"):
        tmax = args['threadsMax']
        for t in range(1, tmax+1):
            args2['threads'] = t
            c.append(command(codePar, args2))

    if args['version'] == "PoissonMPI":
        for n in range(1, nmax+1):
            prefix2 = prefix + ['-np', str(n)]
            c.append(command(codePar, args2, prefix2))

    if args['version'] == "PoissonMPI_OpenMP":
        prefix2 = prefix + ['-np', '1']
        args2['threads'] = 1
        c.append(command(codePar, args2, prefix2))
        
        nmax = args['npMax']
        n = 1
        t = nmax
        while n <= nmax:
            prefix2 = prefix + ['-np', str(n)]
            args2['threads'] = t
            c.append(command(codePar, args2, prefix2))
            n *= 2
            t = int(t/2)

    return c

def times_omp(args, log):
    tmax = args['threadsMax']

    u0,v0 = readTemps()

    if tmax > 1:
        t = []
        x = []
        last = []
        speedup = []

        for i in range(1, tmax+1):
            u, v = readTemps(t=i)
            t.append(i)
            x.append(u)
            last.append(v)
            if not u0:
                u0 = u
            speedup.append(u0/u)

        if v0:
            s = 'last diff (sequential) = ' + "{:12.3f}".format(v0) + "\n\n"
            sys.stdout.write(s)
            log.write(s)

        s = "threads:  " +  "".join(["{:12d}".format(u) for u in t]) + "\n"
        sys.stdout.write(s)
        log.write(s)
    
        if speedup:
            s = "speedups: " + "".join(["{:12.3f}".format(u) for u in speedup]) + "\n"
            sys.stdout.write(s)
            log.write(s)
    
        s = "last diff:" + "".join(["{:12.3f}".format(u) for u in last]) + "\n\n"
        sys.stdout.write(s)
        log.write(s)

        plot(t, x, speedup, args)

def times_mpi(args, log=None):
    nmax = args['npMax']

    t = []
    x = []
    last = []
    speedup = []
    u0 = None
    v0 = None

    for i in range(1, nmax+1):
        u, v = readTemps(n=i)
        t.append(i)
        x.append(u)
        last.append(v)
        if not u0:
            u0 = u
            v0 = v
        speedup.append(u0/u)

    output = ''
    if v0:
        output += 'last diff (sequential) = ' + "{:12.3f}".format(v0) + "\n\n"
        output += 'processes:' +  "".join(["{:12d}".format(u) for u in t]) + "\n"
        output += 'speedups: ' + "".join(["{:12.3f}".format(u) for u in speedup]) + "\n"
        output += 'last diff:' + "".join(["{:12.3f}".format(u) for u in last]) + "\n\n"

        sys.stdout.write(output)
        if log:
           log.write(output)
        plot(t, x, speedup, args)
    
def plot(t, x, speedup, args):
    try:
        import matplotlib
    except:
        return
    if not args['graphics']:
        matplotlib.use('Agg')
        display = False
    else:
        display = True

    import matplotlib.pyplot as plt

    if speedup:
        nGraphs = 2
    else:
        nGraphs = 1

    fig, (ax1, ax2) = plt.subplots(nGraphs, 1, figsize=(10,16))

    if args['version'].startswith('PoissonOpenMP'):
        xmax = args['threadsMax']
        xlabel = "Threads"
    elif args['version'] == 'PoissonMPI':
        xmax = args['npMax']
        xlabel = "Processes"
    
    ax1.plot(t, x, 'o-')
    ax1.set_ylim(bottom=0.0, top=max(x)*1.1)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel('Temps CPU (s)')
    ax1.xaxis.set_ticks(range(1, xmax+1))
    ax1.grid()

    ax2.plot(t, speedup, 'o-')
    ax2.plot(t, t, '-')
    ax2.set_ylim(bottom=0.0, top=xmax+1)
    ax2.legend([unicode('mesure'), unicode('ideal')])
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel('Speedup')
    ax2.xaxis.set_ticks(range(1,xmax+1))
    ax2.grid()

    fig.set_size_inches(6, 7)
    plt.savefig("speedups_" + args['version'] + ".pdf")

    try:
        if display:
            plt.show()
    except:
        pass
        

