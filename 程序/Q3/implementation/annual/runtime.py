"""Durable bounded resources and content-addressed, completed-history-only bases."""
import sys, os, json, gzip, time, hashlib, copy
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from core import serial,digest
from forecasting import base_predict,array_hash

def encoded(value): return json.dumps(value,ensure_ascii=False,default=serial,separators=(',',':'),allow_nan=False).encode('utf-8')
def read(path):
    p=Path(path)
    return json.loads(gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def atomic(path,value,payload=None):
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_name(p.name+'.tmp')
    data=payload if payload is not None else encoded(value)
    if payload is None and p.suffix=='.gz': data=gzip.compress(data,compresslevel=1,mtime=0)
    with tmp.open('wb') as f: f.write(data); f.flush(); os.fsync(f.fileno())
    # Windows readers/indexers may briefly deny replacing an otherwise valid
    # file. Retry only the identical durable payload, never solver work.
    for attempt in range(6):
        try:
            os.replace(tmp,p)
            break
        except PermissionError:
            if attempt==5: raise
            time.sleep(.02*(2**attempt))
    return hashlib.sha256(data).hexdigest()

class StopRun(RuntimeError): pass

class Budget:
    def __init__(self,out,limits,start_epoch=None,prior_wall=0):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True); self.limits=limits
        self.start=time.perf_counter(); self.offset=max(0,time.time()-(start_epoch or time.time()))
        self.prior_wall=prior_wall; self.calls=0; self.fits=0; self.timings={}
        self.events_path=self.out/'resource_events.jsonl'
        if self.events_path.exists():
            for line in self.events_path.read_text(encoding='utf-8').splitlines():
                e=json.loads(line); self.fits+=e['kind']=='fit'; self.calls+=e['kind']=='lp'
        if (self.out/'resource_progress.json').exists(): self.timings=read(self.out/'resource_progress.json').get('timings',{})
        self.bytes=sum(p.stat().st_size for p in self.out.rglob('*') if p.is_file())
    def elapsed(self): return self.prior_wall+self.offset+time.perf_counter()-self.start
    def remaining(self): return self.limits['wall_seconds']-self.elapsed()
    def check(self):
        if self.remaining()<=0: raise StopRun('Cumulative wall budget exhausted')
        if self.bytes>self.limits['output_bytes']: raise StopRun('Output budget exhausted')
    def write(self,path,value):
        self.check(); p=Path(path); old=p.stat().st_size if p.exists() else 0
        tick=time.perf_counter()
        payload=encoded(value)
        if p.suffix=='.gz': payload=gzip.compress(payload,compresslevel=1,mtime=0)
        if self.bytes+len(payload)-old>self.limits['output_bytes']: raise StopRun('Next output exceeds disk budget')
        h=atomic(p,value,payload); self.bytes+=p.stat().st_size-old
        self.addtime('persistence',time.perf_counter()-tick); self.check(); return h
    def reserve(self,kind,label):
        self.check()
        if kind not in ('lp','fit'): raise ValueError('Annual accepts LP and base fit only')
        if (self.fits if kind=='fit' else self.calls)>=self.limits['fits' if kind=='fit' else 'calls']: raise StopRun(kind+' count budget exhausted')
        event=encoded(dict(kind=kind,label=label,elapsed=self.elapsed(),sequence=self.calls+self.fits+1))+b'\n'
        with self.events_path.open('ab') as f: f.write(event); f.flush(); os.fsync(f.fileno())
        self.bytes+=len(event)
        if kind=='fit': self.fits+=1
        else: self.calls+=1
        self.write(self.out/'resource_progress.json',self.record())
    def addtime(self,key,seconds): self.timings[key]=self.timings.get(key,0)+seconds
    def record(self): return dict(optimizer_calls=self.calls,fits=self.fits,wall_seconds=self.elapsed(),timings=self.timings,limits=self.limits,output_bytes=self.bytes)

class BaseCache:
    def __init__(self,folder,identity,budget):
        self.folder=Path(folder); self.identity=identity; self.budget=budget; self.hot={}
    def get(self,access,day):
        if access.position!=day*144: raise ValueError('Midnight cache requested at wrong cutoff')
        key=digest(dict(source=self.identity['source_sha256'],model=self.identity['prediction_sha256'],day=day,cutoff=day*144,leads=list(range(1 if day==364 else 2))))
        path=self.folder/(key+'.json.gz'); hit=path.exists()
        tick=time.perf_counter()
        # Even hits obtain guarded historical hashes; never share official/SOC/orders.
        histories={v:access.history_before(day,v) for v in ('load','pv')}
        hashes={v:array_hash(a) for v,a in histories.items()}
        if hit:
            obj=read(path)
            if obj['key']!=key or obj['history_hashes']!=hashes or obj['identity']!=self.identity: raise ValueError('Cache identity mismatch')
        else:
            tasks=[]; predictions={'load':[],'pv':[]}
            for k in range(1 if day==364 else 2):
                for v in ('load','pv'):
                    y,record=base_predict(histories[v],day,k,v=='load',self.budget)
                    tasks.append(dict(variable=v,lead=k,target_day=day+k,issued_position=day*144,history_cutoff=day*144,record=record))
                    predictions[v].extend(y.tolist())
            obj=dict(key=key,identity=self.identity,day=day,history_cutoff=day*144,history_hashes=hashes,tasks=tasks,predictions=predictions)
            obj=json.loads(encoded(obj))
            self.budget.write(path,obj)
        self.hot={key:obj}
        access._event('base_cache',key=key,hit=hit,day=day,history_cutoff=day*144,cache_sha256=sha(path))
        self.budget.addtime('base_cache',time.perf_counter()-tick)
        return copy.deepcopy(obj)
