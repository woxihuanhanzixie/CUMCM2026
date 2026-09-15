import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
os.environ['OPENBLAS_NUM_THREADS']='1'
sys.dont_write_bytecode=True
if os.name=='nt' and (Path(sys.executable).parent/'Library/bin').is_dir():
    _dll=os.add_dll_directory(str(Path(sys.executable).parent/'Library/bin'))
