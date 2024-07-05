#!  /usr/bin/env python3

import os, sys, subprocess, argparse, platform, shutil

def manage():
  p = platform.system()
  if p == 'Windows':
      defaultCompiler = 'intel'
  else:
      defaultCompiler = 'gnu'

  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--compiler', default=defaultCompiler,
                      choices=['gnu', 'intel'])
  parser.add_argument('-t', '--type', default='Release',
                      choices=['Debug', 'Release', 'RelWithDebInfo'])
  args = parser.parse_args()

  myenv = os.environ.copy()
  if args.compiler == 'gnu':
    gen = '-GUnix Makefiles'
    myenv['CC'] = 'gcc'
    myenv['CXX'] = 'g++'
    compileCmd = ['make', '--no-print-directory', 'install']
  elif args.compiler == 'intel':
    if p == 'Windows':
      gen = '-GNinja'
      myenv['CC'] = 'icx'
      myenv['CXX'] = 'icx'
      compileCmd = ['ninja', 'install']
    else:
      gen = '-GUnix Makefiles'
      if shutil.which('icpx'):
        myenv['CC'] = 'icx'
        myenv['CXX'] = 'icpx'
      else:
        myenv['CC'] = 'icc'
        myenv['CXX'] = 'icpc'
      compileCmd = ['make', '--no-print-directory', 'install', 'VERBOSE=1']

  baseDir = os.getcwd()
  srcDir = os.path.join(baseDir, 'src')

  print (' '.join(['\nbuild ', args.type, '\n']), file=sys.stderr)
  buildDir = os.path.join(baseDir, 'build')
  installDir = os.path.join(baseDir, 'install')

  cmake_params = ['-DCMAKE_BUILD_TYPE=' + args.type]
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)
  cmake_params.append(gen)

  if not os.path.exists(buildDir):
    os.makedirs(buildDir)

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  err = subprocess.call(configureCmd, cwd=buildDir, env=myenv)
  if err == 0:
    err = subprocess.call(compileCmd, cwd=buildDir, env=myenv)

if __name__ == '__main__':
    manage()

