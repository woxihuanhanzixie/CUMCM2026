import runtime
import argparse,json
from workflow import execute
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--days',type=int,default=2);p.add_argument('--resume',action='store_true')
    p.add_argument('--wall-seconds',type=float,default=300);p.add_argument('--annual-authorized',action='store_true')
    a=p.parse_args();print(json.dumps(execute(a.days,a.resume,a.wall_seconds,a.annual_authorized)))
