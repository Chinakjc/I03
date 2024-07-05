#! /usr/bin/env python3

import os, sys, subprocess, platform, argparse

def manage(args):

  if args.debug:
     mode = "debug"
  else:
     mode = "release"

  e = os.environ.copy()
  if args.compiler == 'gnu':
    e['CC'] = 'gcc'
    e['CXX'] = 'g++'
  elif args.compiler == 'intel':
    e['CC'] = 'icx'
    e['CXX'] = 'icpx'
    
  p = platform.system()
  if p == 'Windows':
      defaultCompiler = 'intel'
      gen = '-GNinja'
  else:
      defaultCompiler = 'gnu'
      gen = '-GUnix Makefiles'

  compileCmd = ['make', '--no-print-directory', 'install']

  baseDir = os.getcwd()
  srcDir = os.path.join(baseDir, 'src')

  buildDir = os.path.join(baseDir, 'build', mode)
  installDir = os.path.join(baseDir, 'install', mode)
  if not os.path.exists(buildDir):
    os.makedirs(buildDir)
  
  cmake_params = [gen]
  cmake_params.append('-DCMAKE_INSTALL_PREFIX=' + installDir)
  cmake_params.append('-DCMAKE_BUILD_TYPE=' + mode.capitalize())

  configureCmd = ['cmake'] + cmake_params + [srcDir]
  err = subprocess.call(configureCmd, cwd=buildDir, env=e)
  if err == 0:
    err = subprocess.call(compileCmd, cwd=buildDir, env=e)

if __name__ == '__main__':

  p = platform.system()
  if p == 'Windows':
      defaultCompiler = 'intel'
  else:
      defaultCompiler = 'gnu'

  parser = argparse.ArgumentParser()
  parser.add_argument('-d', '--debug', action='store_true')
  parser.add_argument('-c', '--compiler', default=defaultCompiler)
    
  args = parser.parse_args()

  manage(args)


