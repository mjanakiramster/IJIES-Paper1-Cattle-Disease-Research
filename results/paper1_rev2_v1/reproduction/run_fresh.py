import nbformat, time, os
from nbclient import NotebookClient
os.chdir('/home/claude/fresh/proj'); t = time.time()
nb = nbformat.read('IJIES_Paper1_Experiments_22sections_unexecuted_v2.ipynb', as_version=4)
try:
    NotebookClient(nb, timeout=None, kernel_name='python3', resources={'metadata': {'path': '.'}}).execute()
    print('OK executed in', round((time.time() - t) / 60, 1), 'min', flush=True)
except Exception as e:
    print('FAILED after', round((time.time() - t) / 60, 1), 'min:', repr(e)[:2000], flush=True)
finally:
    nbformat.write(nb, 'IJIES_Paper1_Experiments_22sections_fresh_executed.ipynb')
