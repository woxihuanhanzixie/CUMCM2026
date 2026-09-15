"""Read-only raw data warehouse; controller receives guarded completed views."""
from pathlib import Path
from datetime import datetime,time as clocktime
import io,hashlib,ctypes,msvcrt,os
import numpy as np
from openpyxl import load_workbook
from forecasting import array_hash

def shared_bytes(path):
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateFileW.argtypes=[ctypes.c_wchar_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p,ctypes.c_uint32,ctypes.c_uint32,ctypes.c_void_p]; kernel.CreateFileW.restype=ctypes.c_void_p
    h=kernel.CreateFileW(str(path),0x80000000,7,None,3,0x80,None)
    if h==ctypes.c_void_p(-1).value: raise ctypes.WinError(ctypes.get_last_error())
    with os.fdopen(msvcrt.open_osfhandle(h,os.O_RDONLY|os.O_BINARY),'rb') as f:return f.read()

def minutes(value):
    if isinstance(value,clocktime): return value.hour*60+value.minute
    value=str(value).strip()
    if value=='0:00+1': return 1440
    parts=value.split(':'); return int(parts[0])*60+int(parts[1])

def read_inputs(root,protocol):
    raw={}; hashes={}
    for rel,expected in protocol['source_sha256'].items():
        value=shared_bytes(Path(root)/rel); actual=hashlib.sha256(value).hexdigest()
        if actual!=expected: raise ValueError('Input identity changed: '+rel)
        hashes[rel]=actual
        if rel.endswith('.xlsx'): raw[rel]=value
    wb=load_workbook(io.BytesIO(raw['C题/附件/附件1.xlsx']),read_only=True,data_only=True)
    rows=list(wb.active.values); head=rows[0]; price_col=head.index('电价')
    if [minutes(r[0]) for r in rows[1:]]!=list(range(10,1441,10)): raise ValueError('Price times invalid')
    prices=np.array([r[price_col] for r in rows[1:]],dtype=float); wb.close()
    if not np.isfinite(prices).all() or (prices<=0).any(): raise ValueError('Invalid prices')
    # Attachment1 nonprice columns never enter the returned interface.
    wb=load_workbook(io.BytesIO(raw['C题/附件/附件2.xlsx']),read_only=True,data_only=True); series=[]
    for name in ('小区负载','光伏发电实际功率'):
        rows=list(wb[name].values)
        if [minutes(v) for v in rows[0][1:]]!=list(range(10,1441,10)): raise ValueError('Actual times invalid')
        if [(r[0].date()-datetime(2025,1,1).date()).days for r in rows[1:]]!=list(range(365)): raise ValueError('Actual date coverage')
        a=np.array([r[1:] for r in rows[1:]],dtype=float).ravel()
        if a.shape!=(52560,) or not np.isfinite(a).all() or (a<0).any(): raise ValueError('Actual values invalid')
        series.append(a)
    wb.close()
    wb=load_workbook(io.BytesIO(raw['C题/附件/附件3.xlsx']),read_only=True,data_only=True); rows=list(wb.active.values); official={}; day=None
    if list(rows[0][2:])!=[f'预报{h}小时' for h in range(1,25)]: raise ValueError('Official columns')
    for row in rows[1:]:
        if row[0] is not None and str(row[0]).strip(): day=(datetime.strptime(str(row[0]).split()[0],'%Y-%m-%d')-datetime(2025,1,1)).days
        slot=minutes(row[1])//10; key=(day,slot)
        a=np.array(row[2:],dtype=float)
        if key in official or slot not in (0,36,72,108) or a.shape!=(24,) or not np.isfinite(a).all() or (a<0).any(): raise ValueError('Invalid official release')
        official[key]=a
    wb.close()
    if set(official)!={(d,h) for d in range(365) for h in (0,36,72,108)}: raise ValueError('Official coverage')
    return dict(load_nodes=series[0],pv_nodes=series[1],prices=np.r_[prices[-1],prices[:-1]],official=official,hashes=hashes)

class Access:
    def __init__(self,data,policy,position=0):
        if policy not in ('C0','C1','C2'): raise ValueError('Unknown policy')
        self.__load=np.asarray(data['load_nodes']).copy(); self.__pv=np.asarray(data['pv_nodes']).copy()
        self.__official={k:np.asarray(v).copy() for k,v in data['official'].items()}
        self.prices=np.asarray(data['prices']).copy(); self.policy=policy; self.position=position; self.committed=None; self.log=[]
    def _event(self,kind,**kw): self.log.append(dict(kind=kind,position=self.position,**kw))
    def _past(self,end,variable):
        if not 0<=end<=self.position: self._event('DENIED',requested_end=end); raise ValueError('Future actual access')
        raw=self.__load if variable=='load' else self.__pv
        ids=np.maximum(np.arange(end)-1,0)
        return raw[ids].copy()
    def history_before(self,day,variable):
        if day*144>self.position: raise ValueError('Future midnight history')
        a=self._past(day*144,variable).reshape(day,144); self._event('history',day=day,variable=variable,sha256=array_hash(a)); return a
    def observations_completed_by(self,end,variable):
        a=self._past(end,variable); self._event('completed',end=end,variable=variable,sha256=array_hash(a)); return a
    def official_released_by(self,day,slot):
        if day*144+slot>self.position or slot not in (0,36,72,108) or (self.policy=='C0' and slot!=0): self._event('DENIED_OFFICIAL',day=day,slot=slot); raise ValueError('Unauthorized official release')
        a=self.__official[(day,slot)].copy(); self._event('official',day=day,slot=slot,sha256=array_hash(a)); return a
    def commit(self,position,order):
        if position!=self.position or self.committed is not None or not np.isfinite(order) or order<0: raise ValueError('Invalid current commitment')
        self.committed=float(order); self._event('commit',order=order)
    def reveal_current_after_commit(self,position):
        if position!=self.position or self.committed is None: raise ValueError('Actual revealed before order commitment')
        source=max(position-1,0)
        self._event('reveal',source_node_index=source,approximation=position==0)
        return float(self.__load[source]),float(self.__pv[source])
    def complete(self):
        if self.committed is None: raise ValueError('Complete without commitment')
        self.position+=1; self.committed=None
