Exemple sur la  windows cmd

call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"

python build.py

python run.py --threadsMax 8 --balances 5

python plot.py


Les résultats correspondent aux dossiers de t[n]b[m], où n et m sont respectivement le nombre de threads cpu et la période de balance. (si m = 0, il n'y a pas de balance)