"""Initialize native library search paths before importing numerical packages."""
import os
import sys
from pathlib import Path

_dll_handles = []
if sys.platform == 'win32':
    for path in (Path(sys.prefix) / 'Library' / 'bin', Path(sys.prefix) / 'DLLs'):
        if path.is_dir():
            _dll_handles.append(os.add_dll_directory(str(path)))
# Many tiny ridge systems do not benefit from a large BLAS thread pool.
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
