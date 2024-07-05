#! /usr/bin/env python3

import glob, os, sys
try:
  import matplotlib
  use_graphics = True
except:
  use_graphics = False

try:
  import numpy
  use_numpy = True
except:
  use_numpy = False

def collectThreads(version):
   listFiles = glob.glob(os.path.join('results', 'temps_*'))
   refFile = os.path.join('results', 'temps.dat')
   if len(listFiles) == 0:
     return None

   listFiles.sort()

   nf = len(listFiles)

   if use_numpy:
     th = numpy.zeros(nf, dtype=int)
     t = numpy.zeros(nf)
     r = numpy.zeros(nf)
     speedup = numpy.zeros(nf)
   else:
     th = []
     t = []
     r = []
     speedup = []
     for i in range(nf):
         th.append(0)
         t.append(0.0)
         r.append(0.0)
         speedup.append(0.0)

   tRef =  None
   if os.path.exists(refFile):
      with open(refFile) as fIn:
        l = fIn.readline().split()
        tRef = float(l[1])
        rRef = float(l[3])
      
   for f in listFiles:
     with open(f) as fIn:
       l = fIn.readline().split()
       i = int(l[0])-1
       th[i] = i+1
       t[i] = float(l[1])
       r[i] = float(l[3])

   if not tRef:
      tRef = t[0]
      rRef = r[0]

   for i in range(len(t)):
      speedup[i] = tRef/t[i]

   s = '\nlast diff (sequential) = ' + "{:12.3f}".format(rRef) + "\n\n"
   s += "threads   :" + "".join(["{:12d}".format(u) for u in th]) + "\n"
   s += "temps     :" + "".join(["{:12.3f}".format(u) for u in t]) + "\n"
   s += "speedups  :" + "".join(["{:12.3f}".format(u) for u in speedup]) + "\n"
   s += "last diff :" + "".join(["{:12.3f}".format(u) for u in r]) + "\n\n"
   print(s)

   with open('log_' + version + '.txt', 'a') as fOut:
      fOut.write(s)

   return (th, t, speedup)

def collectProcesses(version):
   listFiles = glob.glob(os.path.join('results', 'temps_*'))
   if len(listFiles) == 0:
     return None

   listFiles.sort()

   nf = len(listFiles)

   if use_numpy:
     np = numpy.zeros(nf, dtype=int)
     t = numpy.zeros(nf)
     r = numpy.zeros(nf)
     speedup = numpy.zeros(nf)
   else:
     np = []
     t = []
     r = []
     speedup = []
     for i in range(nf):
         np.append(0)
         t.append(0.0)
         r.append(0.0)
         speedup.append(0.0)

   for f in listFiles:
     with open(f) as fIn:
       l = fIn.readline().split()
       i = int(l[0])-1
       np[i] = i+1
       t[i] = float(l[1])
       r[i] = float(l[3])

   for i in range(len(t)):
      speedup[i] = t[0]/t[i]

   s = "\n"
   s += "processes :" +  "".join(["{:12d}".format(u) for u in np]) + "\n"
   s += "temps     :" + "".join(["{:12.3f}".format(u) for u in t]) + "\n"
   s += "speedups  :" + "".join(["{:12.3f}".format(u) for u in speedup]) + "\n"
   s += "last diff :" + "".join(["{:12.3f}".format(u) for u in r]) + "\n\n"
   print(s)

   with open('log_' + version + '.txt', 'w') as fOut:
      fOut.write(s)

   return (np, t, speedup)

def collectHybrid(version):
   listFiles = glob.glob(os.path.join('results', 'temps_*'))
   if len(listFiles) == 0:
     return None

   listFiles.sort()

   nf = len(listFiles)

   if use_numpy:
     np = numpy.zeros(nf, dtype=int)
     th = numpy.zeros(nf, dtype=int)
     t = numpy.zeros(nf)
     r = numpy.zeros(nf)
     speedup = numpy.zeros(nf)
   else:
     np = []
     th = []
     t = []
     r = []
     speedup = []
     for i in range(nf):
         np.append(0)
         th.append(0)
         t.append(0.0)
         r.append(0.0)
         speedup.append(0.0)

   i = 0
   for f in listFiles:
     with open(f) as fIn:
       l = fIn.readline().split()
       np[i] = int(l[0])
       th[i] = int(l[1])
       t[i] = float(l[2])
       r[i] = float(l[4])
       i+=1

   for i in range(len(t)):
      speedup[i] = t[0]/t[i]

   s = "\n"
   s += "processes :" +  "".join(["{:12d}".format(u) for u in np]) + "\n"
   s += "threads   :" + "".join(["{:12d}".format(u) for u in th]) + "\n"
   s += "temps     :" + "".join(["{:12.3f}".format(u) for u in t]) + "\n"
   s += "speedups  :" + "".join(["{:12.3f}".format(u) for u in speedup]) + "\n"
   s += "last diff :" + "".join(["{:12.3f}".format(u) for u in r]) + "\n\n"
   print(s)

   with open('log_' + version + '.txt', 'a') as fOut:
      fOut.write(s)

   return (np, th, t, speedup)

def plotHybrid(np, th, t, speedup, version):
   try:
     import matplotlib.pyplot as plt
   except:
      return

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,16))

   xmax = np.size
   tmax = t.max()
   xlabel = "Processes x Threads"
   xticks = range(1, xmax+1)
   xlabels = [ "{:d}x{:d}".format(u,v) for (u,v) in zip(np, th)]
   ax1.bar(xticks, height=t, width=0.5)
   ax1.set_xlim(left=0.5, right=float(xmax+0.5))
   ax1.set_ylim(bottom=0.0, top=tmax*1.1)
   ax1.set_xlabel(xlabel)
   ax1.set_ylabel('CPU Time (s)')
   ax1.xaxis.set_ticks(xticks)
   ax1.xaxis.set_ticklabels(xlabels)
   ax1.grid()

   ax2.plot(xticks, speedup, 'o')
   ax2.set_xlim(left=0.5, right=float(xmax+0.5))
   ax2.set_ylim(bottom=0.0, top=speedup.max()*1.1)
   for i in range(xmax):
     ax2.annotate("{:5.2f}".format(speedup[i]), (xticks[i]+0.05, speedup[i]-0.2))
   ax2.set_xlabel(xlabel)
   ax2.set_ylabel('Speedup')
   ax2.xaxis.set_ticks(xticks)
   ax2.xaxis.set_ticklabels(xlabels)
   ax2.grid()

   fig.set_size_inches(6, 7)
   plt.savefig("speedups_" + version + ".pdf")


def plotThreads(th, t, speedup, version):
   try:
     import matplotlib.pyplot as plt
   except:
      return

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,16))

   xmax = th.size
   tmax = t.max()
   xlabel = "Threads"
   xticks = range(1, xmax+1)
   xlabels = th
   ax1.plot(xticks, t, "o-")
   ax1.set_xlim(left=0.5, right=float(xmax+0.5))
   ax1.set_ylim(bottom=0.0, top=tmax*1.1)
   ax1.set_xlabel(xlabel)
   ax1.set_ylabel('CPU Time (s)')
   ax1.xaxis.set_ticks(xticks)
   ax1.grid()

   ax2.plot(xticks, speedup, 'o-')
   ax2.plot(xticks, xticks, '-')
   ax2.legend(['measured', 'ideal'])
   ax2.set_xlim(left=0.5, right=float(xmax+0.5))
   ax2.set_ylim(bottom=0.0, top=xmax*1.1)
   ax2.set_xlabel(xlabel)
   ax2.set_ylabel('Speedup')
   ax2.xaxis.set_ticks(xticks)
   ax2.grid()

   fig.set_size_inches(6, 7)
   plt.savefig("speedups_" + version + ".pdf")


def plotProcesses(np, t, speedup, version):
   try:
     import matplotlib.pyplot as plt
   except:
      return

   fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10,16))

   xmax = np.size
   tmax = t.max()
   xlabel = "Processes"
   xticks = range(1, xmax+1)
   xlabels = [ "{:d}".format(u) for u in np]
   ax1.plot(xticks, t, 'o-')
   ax1.set_xlim(left=0.5, right=float(xmax+0.5))
   ax1.set_ylim(bottom=0.0, top=tmax*1.1)
   ax1.set_xlabel(xlabel)
   ax1.set_ylabel('CPU Time (s)')
   ax1.xaxis.set_ticks(xticks)
   ax1.xaxis.set_ticklabels(xlabels)
   ax1.grid()

   ax2.plot(xticks, speedup, '-o')
   ax2.plot(xticks, xticks, '-')
   ax2.legend(['measured', 'ideal'])
   ax2.set_xlim(left=0.5, right=float(xmax+0.5))
   ax2.set_ylim(bottom=0.0, top=xticks[-1]*1.1)
   ax2.set_xlabel(xlabel)
   ax2.set_ylabel('Speedup')
   ax2.xaxis.set_ticks(xticks)
   ax2.xaxis.set_ticklabels(xlabels)
   ax2.grid()

   fig.set_size_inches(6, 7)
   plt.savefig("speedups_" + version + ".pdf")

baseDir = os.getcwd()
version = os.path.basename(baseDir)

if version.startswith('PoissonOpenMP'):
   (th, t, speedup) = collectThreads(version)
   if use_graphics:
       plotThreads(th, t, speedup, version)
elif version == "PoissonMPI":
   (np, t, speedup) = collectProcesses(version)
   if use_graphics:
       plotProcesses(np, t, speedup, version)
elif version == "PoissonMPI_OpenMP":
   (np, th, t, speedup) = collectHybrid(version)
   if use_graphics:
       plotHybrid(np, th, t, speedup, version)

