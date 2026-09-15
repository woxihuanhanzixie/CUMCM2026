"""Export the archived, reviewed Q43 answer without recalculation or precision loss."""
from pathlib import Path
import argparse,shutil,hashlib
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'答案/result4-3.xlsx').is_file())
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise FileExistsError('Use a new filename')
 a.output.parent.mkdir(parents=True,exist_ok=True);src=ROOT/'答案/result4-3.xlsx';shutil.copy2(src,a.output)
 assert hashlib.sha256(src.read_bytes()).digest()==hashlib.sha256(a.output.read_bytes()).digest()
 print('Archived reviewed answer copied:',a.output)
