#!/usr/bin/env python3
"""Reproducible top-8 black-box optimizer benchmark over 33 classical, shifted, and rotated function families.

The suite uses the released ABG-v30 implementation and its hard-budget objective oracle.
It uses paired optimizer seeds on identical per-trial function instances, equal
objective-evaluation budgets, pre-test fixed metric scales, raw CSVs, paired tests,
Holm correction, benchmark landscape figures, performance profiles, and traces.
"""
from __future__ import annotations
import csv, json, math, os, sys, time, zlib, platform, warnings
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
import scipy
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

ROOT = Path(__file__).resolve().parent
ABG_PATH = ROOT / 'ABG_research_v30.py'
OUT = ROOT / 'ABG_top8_benchmark'
PLOTS = OUT / 'plots'
LANDSCAPES = PLOTS / 'landscapes'
OUT.mkdir(parents=True, exist_ok=True); PLOTS.mkdir(exist_ok=True); LANDSCAPES.mkdir(exist_ok=True)

spec = importlib.util.spec_from_file_location('abg_v30_benchmark_source', ABG_PATH)
abg = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = abg
spec.loader.exec_module(abg)

# Add Adam with an explicit fixed learning rate: no claim of task-specific tuning.
abg.METHOD_REGISTRY['Adam (finite-difference, lr=0.03)'] = (
    lambda p, b, s, n, lr=None: abg.run_adam(p, b, s, n, 0.03)
)
METHODS = [
    'ABG-v30',
    'DIRECT',
    'CMA-ES (compact implementation)',
    'Differential Evolution',
    'Dual Annealing',
    'Sobol+L-BFGS-B',
    'Powell multi-start',
    'Adam (finite-difference, lr=0.03)',
]
# Remove methods unavailable in this SciPy version rather than silently fabricating results.
METHODS = [m for m in METHODS if m in abg.METHOD_REGISTRY]
BUDGET = 300
TRIALS = 10
DIMS = (2, 10, 30)
NOISE_FUNCTIONS = ['Sphere (shifted convex)', 'Rotated Ellipsoid (convex)', 'Rosenbrock', 'Shifted Rastrigin', 'Ackley', 'Schwefel']
NOISE_LEVELS = (0.01, 0.05)
NOISE_TRIALS = 6

# ---------- benchmark function definitions ----------
def fn_ellipsoid(x):
    x=np.asarray(x,float); weights=np.power(10.0, 6.0*np.linspace(0,1,x.size) if x.size>1 else np.array([0.0]))
    return float(np.sum(weights*x*x))
def fn_bent_cigar(x):
    x=np.asarray(x,float); return float(x[0]**2 + 1e6*np.sum(x[1:]**2))
def fn_discus(x):
    x=np.asarray(x,float); return float(1e6*x[0]**2 + np.sum(x[1:]**2))
def fn_zakharov(x):
    x=np.asarray(x,float); i=np.arange(1,x.size+1,dtype=float); s=np.sum(0.5*i*x)
    return float(np.sum(x*x)+s*s+s**4)
def fn_rosen(x):
    x=np.asarray(x,float); return float(np.sum(100*(x[1:]-x[:-1]**2)**2+(1-x[:-1])**2))
def fn_beale(x):
    x,y=np.asarray(x,float)
    return float((1.5-x+x*y)**2+(2.25-x+x*y*y)**2+(2.625-x+x*y**3)**2)
def fn_himmelblau(x):
    x,y=np.asarray(x,float); return float((x*x+y-11)**2+(x+y*y-7)**2)
def fn_michalewicz_from_z(z):
    # Map normalized domain z in [-pi, pi] to the standard [0, pi] domain.
    u=(np.asarray(z,float)+np.pi)/2.0; i=np.arange(1,u.size+1,dtype=float); m=10
    return float(-np.sum(np.sin(u)*np.sin(i*u*u/np.pi)**(2*m)))
def fn_powell_singular(x):
    x=np.asarray(x,float); total=0.0
    for j in range(0,x.size,4):
        a,b,c,d=x[j:j+4]
        total+=(a+10*b)**2+5*(c-d)**2+(b-2*c)**4+10*(a-d)**4
    return float(total)
def fn_cross_in_tray(x):
    x,y=np.asarray(x,float)
    t=abs(np.sin(x)*np.sin(y)*np.exp(abs(100-np.sqrt(x*x+y*y)/np.pi)))
    return float(-0.0001*(t+1)**0.1)
def fn_eggholder(x):
    x,y=np.asarray(x,float)
    return float(-(y+47)*np.sin(np.sqrt(abs(x/2+y+47))) - x*np.sin(np.sqrt(abs(x-(y+47)))))
def fn_alpine1(x):
    x=np.asarray(x,float); return float(np.sum(np.abs(x*np.sin(x)+0.1*x)))
def fn_dixon_price(x):
    x=np.asarray(x,float); i=np.arange(2,x.size+1,dtype=float)
    return float((x[0]-1)**2+np.sum(i*(2*x[1:]**2-x[:-1])**2))
def fn_levy(x):
    x=np.asarray(x,float); w=1+(x-1)/4
    first=np.sin(np.pi*w[0])**2
    middle=np.sum((w[:-1]-1)**2*(1+10*np.sin(np.pi*w[:-1]+1)**2))
    last=(w[-1]-1)**2*(1+np.sin(2*np.pi*w[-1])**2)
    return float(first+middle+last)
def rotation_matrix(d, seed):
    if d == 1: return np.ones((1,1))
    rng=np.random.default_rng(seed)
    raw=rng.normal(size=(d,d)); q,r=np.linalg.qr(raw)
    signs=np.sign(np.diag(r)); signs[signs==0]=1
    return q*signs[None,:]

# name, category, description, dimension rule
SCALABLE = [
 ('Sphere (convex)', 'Convex', 'Smooth, isotropic quadratic with a known minimum.'),
 ('Sphere (shifted convex)', 'Convex, shifted', 'Convex sphere translated to a different center per trial.'),
 ('Ellipsoid (ill-conditioned convex)', 'Convex, ill-conditioned', 'Separable quadratic with condition number about 10^6.'),
 ('Rotated Ellipsoid (convex)', 'Convex, rotated, ill-conditioned', 'Ill-conditioned quadratic with random orthogonal axes per trial.'),
 ('Bent Cigar (convex)', 'Convex, ill-conditioned', 'One gentle axis and very steep remaining directions.'),
 ('Discus (convex)', 'Convex, ill-conditioned', 'One very steep direction and gentler remaining directions.'),
 ('Zakharov (convex)', 'Convex', 'Quadratic plus convex powers of a linear form.'),
 ('Rosenbrock', 'Nonconvex, narrow valley', 'Classic curved valley with a global minimum at the all-ones vector.'),
 ('Rotated Rosenbrock', 'Nonconvex, rotated valley', 'Rosenbrock valley rotated and translated per trial.'),
 ('Rastrigin', 'Multimodal', 'Periodic landscape with many local minima.'),
 ('Shifted Rastrigin', 'Multimodal, shifted', 'Rastrigin with a noncentral global minimum.'),
 ('Rotated Rastrigin', 'Multimodal, rotated and shifted', 'Rotated Rastrigin with a noncentral global minimum.'),
 ('Ackley', 'Multimodal, flat basin', 'Flat central basin with a rugged outer landscape.'),
 ('Griewank', 'Multimodal', 'Many local minima across a broad domain.'),
 ('Schwefel', 'Multimodal, deceptive', 'Global optimum lies far from the origin.'),
 ('Levy', 'Multimodal', 'Oscillatory landscape with a nonzero global minimizer.'),
 ('Styblinski-Tang', 'Nonconvex, separable', 'Nonconvex quartic with competing minima.'),
 ('Dixon-Price', 'Nonconvex, coupled', 'Coupled polynomial with a nontrivial global minimizer.'),
 ('Alpine 1', 'Nonsmooth, multimodal', 'Nonsmooth oscillatory objective with many stationary features.'),
 ('Shifted Ellipsoid (ill-conditioned convex)', 'Convex, shifted, ill-conditioned', 'Ill-conditioned quadratic shifted away from the origin.'),
 ('Shifted Rotated Ellipsoid (convex)', 'Convex, shifted, rotated, ill-conditioned', 'Randomly rotated and shifted ill-conditioned quadratic.'),
 ('Shifted Bent Cigar (convex)', 'Convex, shifted, ill-conditioned', 'Bent-cigar quadratic shifted away from the origin.'),
 ('Shifted Discus (convex)', 'Convex, shifted, ill-conditioned', 'Discus quadratic shifted away from the origin.'),
 ('Shifted Zakharov (convex)', 'Convex, shifted', 'Zakharov landscape translated away from the origin.'),
 ('Shifted Ackley', 'Multimodal, shifted, flat basin', 'Ackley landscape with a noncentral global minimum.'),
 ('Shifted Griewank', 'Multimodal, shifted', 'Griewank landscape with a noncentral global minimum.'),
 ('Shifted Alpine 1', 'Nonsmooth, shifted multimodal', 'Alpine 1 shifted to remove the central optimum.'),
]
SPECIAL = [
 ('Beale (2D)', 'Multimodal, 2D-only', 'Two-variable polynomial with a known zero minimum.', 2),
 ('Himmelblau (2D)', 'Multimodal, 2D-only', 'Four separated global minima.', 2),
 ('Michalewicz (2D)', 'Multimodal, 2D-only', 'Sharp nonconvex minima; m=10.', 2),
 ('Powell Singular (4D)', 'Nonconvex, ill-conditioned, 4D', 'Classic Powell singular quartic with coupled variables.', 4),
 ('Cross-in-Tray (2D)', 'Highly multimodal, 2D-only', 'Sharp exponential-sine landscape with multiple global minima.', 2),
 ('Eggholder (2D)', 'Highly multimodal, 2D-only', 'Highly rugged 2D landscape with a boundary-near global minimum.', 2),
]
ALL_NAMES=[x[0] for x in SCALABLE]+[x[0] for x in SPECIAL]


def build_problem(name, dim, instance):
    rng=np.random.default_rng(900_000 + 9973*int(instance) + zlib.crc32(name.encode()) + 101*dim)
    scale=1.0; fstar=0.0; opt=np.zeros(dim); R=np.eye(dim); shift=np.zeros(dim); desc=''; category=''; fn=None
    if name=='Sphere (convex)':
        scale=5.0; fn=lambda z: float(np.sum(np.asarray(z)**2)); opt=np.zeros(dim)
    elif name=='Sphere (shifted convex)':
        scale=5.0; shift=scale*rng.uniform(-0.35,0.35,dim); fn=lambda z,s=shift: float(np.sum((np.asarray(z)-s)**2)); opt=shift.copy()
    elif name=='Ellipsoid (ill-conditioned convex)':
        scale=1.0; fn=fn_ellipsoid; opt=np.zeros(dim)
    elif name=='Rotated Ellipsoid (convex)':
        scale=1.0; R=rotation_matrix(dim, 7001+int(instance)*31+dim); fn=lambda z,r=R: fn_ellipsoid(r.T@np.asarray(z)); opt=np.zeros(dim)
    elif name=='Bent Cigar (convex)': scale=1.0; fn=fn_bent_cigar; opt=np.zeros(dim)
    elif name=='Discus (convex)': scale=1.0; fn=fn_discus; opt=np.zeros(dim)
    elif name=='Zakharov (convex)': scale=5.0; fn=fn_zakharov; opt=np.zeros(dim)
    elif name=='Rosenbrock':
        scale=2.048; opt=np.ones(dim); fn=fn_rosen
    elif name=='Rotated Rosenbrock':
        scale=2.048; shift=scale*rng.uniform(-0.18,0.18,dim); R=rotation_matrix(dim, 11003+int(instance)*41+dim)
        fn=lambda z,r=R,s=shift: fn_rosen(r.T@(np.asarray(z)-s)+np.ones(dim)); opt=shift.copy()
    elif name=='Rastrigin': scale=5.12; fn=abg._rastrigin; opt=np.zeros(dim)
    elif name=='Shifted Rastrigin':
        scale=5.12; shift=scale*rng.uniform(-0.32,0.32,dim); fn=lambda z,s=shift: abg._rastrigin(np.asarray(z)-s); opt=shift.copy()
    elif name=='Rotated Rastrigin':
        scale=5.12; shift=scale*rng.uniform(-0.25,0.25,dim); R=rotation_matrix(dim, 13007+int(instance)*43+dim)
        fn=lambda z,r=R,s=shift: abg._rastrigin(r.T@(np.asarray(z)-s)); opt=shift.copy()
    elif name=='Ackley': scale=32.768; fn=abg._ackley; opt=np.zeros(dim)
    elif name=='Griewank': scale=600.0; fn=abg._griewank; opt=np.zeros(dim)
    elif name=='Schwefel': scale=500.0; fn=abg._schwefel; opt=np.full(dim, 420.9687462275036)
    elif name=='Levy': scale=10.0; fn=fn_levy; opt=np.ones(dim)
    elif name=='Styblinski-Tang': scale=5.0; fn=abg._styblinski_tang; opt=np.full(dim,-2.903534027771178); fstar=-39.1661657037714*dim
    elif name=='Dixon-Price':
        scale=10.0; fn=fn_dixon_price; t=np.empty(dim); t[0]=1.0
        for i in range(1,dim): t[i]=math.sqrt(max(t[i-1],0)/2)
        opt=t
    elif name=='Alpine 1': scale=10.0; fn=fn_alpine1; opt=np.zeros(dim)
    elif name=='Shifted Ellipsoid (ill-conditioned convex)':
        scale=1.0; shift=scale*rng.uniform(-0.35,0.35,dim); fn=lambda z,s=shift: fn_ellipsoid(np.asarray(z)-s); opt=shift.copy()
    elif name=='Shifted Rotated Ellipsoid (convex)':
        scale=1.0; shift=scale*rng.uniform(-0.30,0.30,dim); R=rotation_matrix(dim, 17011+int(instance)*47+dim); fn=lambda z,r=R,s=shift: fn_ellipsoid(r.T@(np.asarray(z)-s)); opt=shift.copy()
    elif name=='Shifted Bent Cigar (convex)':
        scale=1.0; shift=scale*rng.uniform(-0.35,0.35,dim); fn=lambda z,s=shift: fn_bent_cigar(np.asarray(z)-s); opt=shift.copy()
    elif name=='Shifted Discus (convex)':
        scale=1.0; shift=scale*rng.uniform(-0.35,0.35,dim); fn=lambda z,s=shift: fn_discus(np.asarray(z)-s); opt=shift.copy()
    elif name=='Shifted Zakharov (convex)':
        scale=5.0; shift=scale*rng.uniform(-0.30,0.30,dim); fn=lambda z,s=shift: fn_zakharov(np.asarray(z)-s); opt=shift.copy()
    elif name=='Shifted Ackley':
        scale=32.768; shift=scale*rng.uniform(-0.25,0.25,dim); fn=lambda z,s=shift: abg._ackley(np.asarray(z)-s); opt=shift.copy()
    elif name=='Shifted Griewank':
        scale=600.0; shift=scale*rng.uniform(-0.25,0.25,dim); fn=lambda z,s=shift: abg._griewank(np.asarray(z)-s); opt=shift.copy()
    elif name=='Shifted Alpine 1':
        scale=10.0; shift=scale*rng.uniform(-0.35,0.35,dim); fn=lambda z,s=shift: fn_alpine1(np.asarray(z)-s); opt=shift.copy()
    elif name=='Beale (2D)': scale=4.5; fn=fn_beale; opt=np.array([3.0,0.5])
    elif name=='Himmelblau (2D)': scale=6.0; fn=fn_himmelblau; opt=np.array([3.0,2.0])
    elif name=='Michalewicz (2D)': scale=np.pi; fn=fn_michalewicz_from_z; opt=2*np.array([2.2029055201,1.5707963268])-np.pi; fstar=-1.801303410098553
    elif name=='Powell Singular (4D)': scale=4.0; fn=fn_powell_singular; opt=np.zeros(dim)
    elif name=='Cross-in-Tray (2D)':
        scale=10.0; shift=rng.uniform(-0.6,0.6,2); fn=lambda z,s=shift: fn_cross_in_tray(np.asarray(z)-s); opt=shift+np.array([1.3494066086,1.3494066086]); fstar=-2.062611870822739
    elif name=='Eggholder (2D)': scale=512.0; fn=fn_eggholder; opt=np.array([512.0,404.231805123]); fstar=-959.6406627106155
    else: raise KeyError(name)
    problem=abg.Problem(name,dim,float(scale),float(fstar),fn,dict([(n,d) for n,c,d in SCALABLE]+[(n,d) for n,c,d,_ in SPECIAL])[name])
    opt_norm=np.asarray(opt,dtype=float)/scale
    # Calibrate a dimensionless reference scale from a method-independent Sobol design.
    ref=abg.sobol_points(256,dim,71_003+zlib.crc32(name.encode())+97*dim+31*int(instance))
    vals=np.asarray([problem.evaluate(x) for x in ref],dtype=float)
    dev=np.abs(vals-fstar); dev=dev[np.isfinite(dev)]
    metric=float(np.median(dev)) if dev.size else 1.0
    if not np.isfinite(metric) or metric<=1e-12: metric=float(np.quantile(dev,0.75)) if dev.size else 1.0
    if not np.isfinite(metric) or metric<=1e-12: metric=1.0
    problem=abg.replace(problem, metric_scale=metric)
    return problem, opt_norm


def score(problem, oracle):
    clean=float(problem.evaluate(oracle.best_x)) if oracle.best_x is not None else float('inf')
    gap=abg.normalized_gap(clean,problem.f_star,problem.metric_scale)
    return clean, gap

RAW_FIELDS=['function','category','dimension','budget','noise_level','noise_sigma','trial','seed','method','best_f_observed','best_f_clean','f_star','metric_scale','gap','success_1e-3','success_1e-6','evals','requests','cache_hits','budget_used_pct','seconds','error','best_x_json','algorithm_data_json','phase_counts_json']
TRACE_FIELDS=['function','dimension','trial','method','eval_index','best_observed','gap_to_best_so_far']

def run_case(problem, name, category, method, budget, noise, trial, seed, trace_on=False):
    t0=time.perf_counter(); error=''; oracle=None
    try:
        oracle=abg.METHOD_REGISTRY[method](problem,budget,seed,noise,None)
        if oracle.evals>budget: raise AssertionError(f'evaluation cap violated: {oracle.evals}>{budget}')
        if sum(oracle.phase_counts.values())!=oracle.evals: raise AssertionError('phase accounting mismatch')
    except Exception as e:
        error=f'{type(e).__name__}: {e}'
    elapsed=time.perf_counter()-t0
    if oracle is None:
        return {'function':name,'category':category,'dimension':problem.dim,'budget':budget,'noise_level':noise,'noise_sigma':noise*problem.metric_scale,'trial':trial,'seed':seed,'method':method,'best_f_observed':float('inf'),'best_f_clean':float('inf'),'f_star':problem.f_star,'metric_scale':problem.metric_scale,'gap':1e30,'success_1e-3':0,'success_1e-6':0,'evals':0,'requests':0,'cache_hits':0,'budget_used_pct':0.0,'seconds':elapsed,'error':error,'best_x_json':'[]','algorithm_data_json':'{}','phase_counts_json':'{}'}, []
    clean,gap=score(problem,oracle)
    row={'function':name,'category':category,'dimension':problem.dim,'budget':budget,'noise_level':noise,'noise_sigma':noise*problem.metric_scale,'trial':trial,'seed':seed,'method':method,'best_f_observed':float(oracle.best_f),'best_f_clean':clean,'f_star':problem.f_star,'metric_scale':problem.metric_scale,'gap':gap,'success_1e-3':int(gap<=1e-3),'success_1e-6':int(gap<=1e-6),'evals':oracle.evals,'requests':oracle.requests,'cache_hits':oracle.cache_hits,'budget_used_pct':100*oracle.evals/max(1,budget),'seconds':elapsed,'error':error,'best_x_json':json.dumps(oracle.best_x.tolist()) if oracle.best_x is not None else '[]','algorithm_data_json':json.dumps(getattr(oracle,'algorithm_data',{}),sort_keys=True,allow_nan=True),'phase_counts_json':json.dumps(oracle.phase_counts,sort_keys=True)}
    trace=[]
    if trace_on and noise==0 and oracle.values:
        vals=np.asarray(oracle.values,float); best=np.minimum.accumulate(vals)
        for j,v in enumerate(best,1):
            g=max(0.0,float(v-problem.f_star))/max(problem.metric_scale,1e-12)
            trace.append({'function':name,'dimension':problem.dim,'trial':trial,'method':method,'eval_index':j,'best_observed':float(v),'gap_to_best_so_far':g})
    return row,trace

def write_rows(path, fields, rows):
    if not rows: return
    exists=path.exists() and path.stat().st_size>0
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        if not exists: w.writeheader()
        w.writerows(rows)

def get_specs():
    specs=[]
    for name,cat,desc in SCALABLE:
        for d in DIMS: specs.append((name,cat,desc,d))
    for name,cat,desc,d in SPECIAL: specs.append((name,cat,desc,d))
    return specs


def run_benchmark(max_new_runs=None):
    raw_path=OUT/'raw_results.csv'; trace_path=OUT/'convergence_traces.csv'; progress_path=OUT/'progress.json'
    specs=get_specs(); det_total=len(specs)*TRIALS*len(METHODS)
    noise_total=len(NOISE_FUNCTIONS)*len(NOISE_LEVELS)*NOISE_TRIALS*len(METHODS)
    trace_names={('Sphere (shifted convex)',2),('Rosenbrock',2),('Rastrigin',2),('Schwefel',10),('Rotated Ellipsoid (convex)',30)}
    existing=set()
    if raw_path.exists() and raw_path.stat().st_size>0:
        old=pd.read_csv(raw_path)
        for r in old[['function','dimension','budget','noise_level','trial','method']].itertuples(index=False,name=None):
            existing.add((str(r[0]),int(r[1]),int(r[2]),round(float(r[3]),8),int(r[4]),str(r[5])))
    done=sum(1 for k in existing if k[3]==0.0)
    ndone=sum(1 for k in existing if k[3]>0.0)
    new_runs=0; start=time.time(); raw_buffer=[]; trace_buffer=[]
    print(f'Resume state: deterministic {done}/{det_total}; noisy {ndone}/{noise_total}; max new runs this pass={max_new_runs}',flush=True)
    for spec_i,(name,cat,desc,d) in enumerate(specs):
        for trial in range(TRIALS):
            problem,opt=build_problem(name,d,trial)
            seed=int(20261019+10007*spec_i+1009*trial+31*d)
            for method in METHODS:
                key=(name,int(d),int(BUDGET),0.0,int(trial),method)
                if key in existing: continue
                if max_new_runs is not None and new_runs>=max_new_runs:
                    if raw_buffer: write_rows(raw_path,RAW_FIELDS,raw_buffer)
                    if trace_buffer: write_rows(trace_path,TRACE_FIELDS,trace_buffer)
                    return False
                do_trace=(name,d) in trace_names
                row,traces=run_case(problem,name,cat,method,BUDGET,0.0,trial,seed,do_trace)
                raw_buffer.append(row); trace_buffer.extend(traces); new_runs+=1; done+=1
                existing.add(key)
                if len(raw_buffer)>=64 or done==det_total:
                    write_rows(raw_path,RAW_FIELDS,raw_buffer); raw_buffer=[]
                    write_rows(trace_path,TRACE_FIELDS,trace_buffer); trace_buffer=[]
                    elapsed=time.time()-start
                    print(f'[deterministic] {done}/{det_total} ({100*done/det_total:.1f}%) elapsed={elapsed:.1f}s latest={name}/d{d}/{method}',flush=True)
                    progress_path.write_text(json.dumps({'deterministic_done':done,'deterministic_total':det_total,'noisy_done':ndone,'noisy_total':noise_total,'new_runs_this_pass':new_runs,'elapsed_sec_this_pass':elapsed,'last_function':name,'last_dim':d,'last_method':method},indent=2),encoding='utf-8')
                elif new_runs % 16 == 0:
                    # lightweight progress without frequent rewriting of the CSV checkpoint
                    print(f'[deterministic] {done}/{det_total} ({100*done/det_total:.1f}%)',flush=True)
    if raw_buffer: write_rows(raw_path,RAW_FIELDS,raw_buffer); raw_buffer=[]
    if trace_buffer: write_rows(trace_path,TRACE_FIELDS,trace_buffer); trace_buffer=[]
    # Noisy stress subset; restart-safe and separately counted.
    if ndone < noise_total:
        noise_buffer=[]
        for fname in NOISE_FUNCTIONS:
            d=10
            cat=dict((n,c) for n,c,desc in SCALABLE)[fname]
            for noise in NOISE_LEVELS:
                for trial in range(NOISE_TRIALS):
                    problem,opt=build_problem(fname,d,trial+101)
                    seed=int(3_100_019+13_007*ALL_NAMES.index(fname)+1009*trial+int(noise*10000))
                    for method in METHODS:
                        key=(fname,int(d),int(BUDGET),round(float(noise),8),int(trial),method)
                        if key in existing: continue
                        if max_new_runs is not None and new_runs>=max_new_runs:
                            if noise_buffer: write_rows(raw_path,RAW_FIELDS,noise_buffer)
                            return False
                        row,_=run_case(problem,fname,cat,method,BUDGET,float(noise),trial,seed,False)
                        noise_buffer.append(row); new_runs+=1; ndone+=1; existing.add(key)
                        if len(noise_buffer)>=64 or ndone==noise_total:
                            write_rows(raw_path,RAW_FIELDS,noise_buffer); noise_buffer=[]
                            print(f'[noise] {ndone}/{noise_total} ({100*ndone/noise_total:.1f}%)',flush=True)
        if noise_buffer: write_rows(raw_path,RAW_FIELDS,noise_buffer)
    print('All optimization runs completed.',flush=True)
    return done>=det_total and ndone>=noise_total

def holm(pvals):
    p=np.asarray([1.0 if not np.isfinite(x) else float(x) for x in pvals]); order=np.argsort(p); out=np.empty_like(p); running=0.
    for rank,idx in enumerate(order):
        running=max(running,min(1.0,(len(p)-rank)*p[idx])); out[idx]=running
    return out.tolist()

def make_summaries():
    df=pd.read_csv(OUT/'raw_results.csv')
    df['gap_safe']=pd.to_numeric(df['gap'],errors='coerce').replace([np.inf,-np.inf],1e30).fillna(1e30).clip(lower=0)
    def group_summary(g):
        x=g['gap_safe'].to_numpy(float)
        return pd.Series({'n_trials':len(g),'median_gap':np.median(x),'mean_gap':np.mean(x),'q1_gap':np.quantile(x,.25),'q3_gap':np.quantile(x,.75),'iqr_gap':np.quantile(x,.75)-np.quantile(x,.25),'success_1e-3_pct':100*g['success_1e-3'].mean(),'success_1e-6_pct':100*g['success_1e-6'].mean(),'median_evals':g['evals'].median(),'median_seconds':g['seconds'].median(),'error_runs':int(g['error'].fillna('').astype(str).ne('').sum())})
    summ=df.groupby(['function','category','dimension','budget','noise_level','method'],sort=False).apply(group_summary,include_groups=False).reset_index()
    summ.to_csv(OUT/'summary_by_function_dimension.csv',index=False)
    # Mean task rank across all tasks; every method is present in all common deterministic/noisy tasks.
    det=summ[summ['noise_level']==0].copy()
    det['rank']=det.groupby(['function','dimension','budget','noise_level'])['median_gap'].rank(method='average',ascending=True)
    overall=det.groupby('method').agg(mean_rank=('rank','mean'),median_rank=('rank','median'),mean_success_1e3=('success_1e-3_pct','mean'),mean_success_1e6=('success_1e-6_pct','mean'),median_seconds=('median_seconds','median'),n_tasks=('rank','count')).reset_index()
    # Task wins, top-3 counts, and Borda score based on paired task summary ranks.
    overall=overall.merge(det.groupby('method').apply(lambda g: pd.Series({'task_wins':int((g['rank']==1).sum()),'top3_tasks':int((g['rank']<=3).sum())}),include_groups=False).reset_index(),on='method')
    overall=overall.sort_values(['mean_rank','mean_success_1e3'],ascending=[True,False])
    overall.to_csv(OUT/'overall_ranking.csv',index=False)
    # Paired Wilcoxon per function/dimension/noise against ABG-v30, Holm correction over all tests.
    tests=[]
    group_cols=['function','dimension','budget','noise_level']
    for keys,g in df.groupby(group_cols,sort=False):
        control=g[g.method=='ABG-v30'].set_index('trial')
        for m in METHODS:
            if m=='ABG-v30': continue
            other=g[g.method==m].set_index('trial')
            trials=sorted(set(control.index)&set(other.index))
            if len(trials)<5: continue
            a=np.maximum(control.loc[trials,'gap_safe'].to_numpy(float),1e-15)
            b=np.maximum(other.loc[trials,'gap_safe'].to_numpy(float),1e-15)
            la=np.log10(a); lb=np.log10(b); delta=lb-la # positive => ABG has smaller gap
            try: p=float(stats.wilcoxon(la,lb,alternative='two-sided',zero_method='pratt').pvalue) if np.any(np.abs(delta)>1e-12) else 1.0
            except ValueError: p=1.0
            tests.append({'function':keys[0],'dimension':keys[1],'budget':keys[2],'noise_level':keys[3],'control':'ABG-v30','competitor':m,'n_pairs':len(trials),'abg_wins':int(np.sum(a<b-1e-12)),'abg_losses':int(np.sum(a>b+1e-12)),'ties':int(np.sum(np.abs(a-b)<=1e-12)),'median_log10_gap_improvement':float(np.median(delta)),'mean_log10_gap_improvement':float(np.mean(delta)),'p_value':p})
    if tests:
        adj=holm([x['p_value'] for x in tests])
        for row,p in zip(tests,adj): row['p_holm_global']=p; row['significant_holm_0_05']=int(p<.05)
    pd.DataFrame(tests).to_csv(OUT/'paired_tests_vs_ABG.csv',index=False)
    return df,summ,overall,tests


def save_landscape_charts():
    for old_png in LANDSCAPES.glob('*.png'):
        old_png.unlink()
    plots_meta=[]
    all_specs=[(n,c,d,2) for n,c,d in SCALABLE]+[(n,c,d,dim) for n,c,d,dim in SPECIAL]
    for k,(name,cat,desc,d) in enumerate(all_specs):
        problem,opt=build_problem(name,d,0)
        n=81; grid=np.linspace(-1,1,n); X,Y=np.meshgrid(grid,grid)
        Z0=np.zeros((n,n)); Zm=np.zeros((n,n))
        # Panel A: global-domain slice with coordinates 3..d fixed at 0.
        # Panel B: slice through the known minimizer with coordinates 3..d fixed at their optimum.
        for iy in range(n):
            for ix in range(n):
                pt0=np.zeros(d); pt0[0]=X[iy,ix]; pt0[1]=Y[iy,ix]
                ptm=np.asarray(opt,dtype=float).copy(); ptm[0]=X[iy,ix]; ptm[1]=Y[iy,ix]
                Z0[iy,ix]=problem.evaluate(pt0)
                Zm[iy,ix]=problem.evaluate(ptm)
        surfaces=[Z0,Zm]
        fig,axes=plt.subplots(1,2,figsize=(13.3,5.8),constrained_layout=True)
        for panel,(ax,Z,label) in enumerate(zip(axes,surfaces,['Other coordinates fixed at 0','Other coordinates fixed at known minimizer'])):
            G=np.maximum(Z-problem.f_star,0.0)
            V=np.log10(1.0+G/max(problem.metric_scale,1e-12))
            vals=V[np.isfinite(V)]; vmax=float(np.quantile(vals,.985)) if vals.size else 1.0
            vmax=max(vmax,1e-9); V=np.clip(V,0,vmax)
            cf=ax.contourf(X,Y,V,levels=24,cmap='viridis',norm=Normalize(0,vmax))
            ax.contour(X,Y,V,levels=8,colors='white',linewidths=.35,alpha=.55)
            if opt.size>=2 and np.all(np.isfinite(opt[:2])) and np.all(np.abs(opt[:2])<=1):
                ax.scatter([opt[0]],[opt[1]],marker='*',s=165,color='red',edgecolor='white',linewidth=.8,label='known global minimizer projection')
                ax.legend(loc='upper right',fontsize=7,framealpha=.8)
            ax.set_xlabel('normalized coordinate x₁ ∈ [-1, 1]'); ax.set_ylabel('normalized coordinate x₂ ∈ [-1, 1]')
            ax.set_title(label,fontsize=9)
            cb=fig.colorbar(cf,ax=ax); cb.set_label('log₁₀(1 + (f(x) − f*) / reference scale)')
        fig.suptitle(f'{name}\n{cat} | 2-D landscape slices',fontsize=11)
        safe=''.join(ch.lower() if ch.isalnum() else '_' for ch in name).strip('_')
        path=LANDSCAPES/f'{k+1:02d}_{safe}.png'; fig.savefig(path,dpi=130); plt.close(fig)
        plots_meta.append({'function':name,'category':cat,'dimension_used_for_plot':d,'file':str(path.relative_to(OUT)),'description':desc})
        print(f'[plots] landscape {k+1}/{len(all_specs)}: {name}',flush=True)
    pd.DataFrame(plots_meta).to_csv(OUT/'landscape_catalog.csv',index=False)

def make_comparison_plots(df,summ,overall,tests):
    # 1 overall ranking
    fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True)
    o=overall.sort_values('mean_rank',ascending=True)
    bars=ax.barh(o['method'],o['mean_rank'])
    ax.invert_yaxis(); ax.set_xlabel('Mean rank across deterministic function × dimension tasks (lower is better)')
    ax.set_title('Top-8 optimizer ranking — median normalized gap; budget=300')
    for b,v in zip(bars,o['mean_rank']): ax.text(v+0.04,b.get_y()+b.get_height()/2,f'{v:.2f}',va='center',fontsize=9)
    fig.savefig(PLOTS/'01_overall_ranking.png',dpi=160); plt.close(fig)
    # 2 success rate by method
    fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True)
    o=overall.sort_values('mean_success_1e3',ascending=True)
    ax.barh(o['method'],o['mean_success_1e3']); ax.set_xlabel('Mean success rate across tasks (%) — normalized gap ≤ 1e-3')
    ax.set_title('Success rate summary (does not replace gap/rank analysis)')
    fig.savefig(PLOTS/'02_success_rate.png',dpi=160); plt.close(fig)
    # 3 median gap heatmap across tasks x algorithms
    det=summ[summ.noise_level==0].copy(); det['task']=det['function']+' / d='+det['dimension'].astype(str)
    piv=det.pivot_table(index='task',columns='method',values='median_gap',aggfunc='median')
    mat=np.log10(np.maximum(piv.to_numpy(float),1e-15))
    fig,ax=plt.subplots(figsize=(14,max(8,0.22*len(piv))),constrained_layout=True)
    im=ax.imshow(mat,aspect='auto',cmap='magma',interpolation='nearest')
    ax.set_xticks(range(len(piv.columns)),piv.columns,rotation=55,ha='right'); ax.set_yticks(range(len(piv.index)),piv.index,fontsize=7)
    ax.set_title('Median normalized optimality gap by task and method (log₁₀ scale)')
    fig.colorbar(im,ax=ax,label='log₁₀ median gap'); fig.savefig(PLOTS/'03_gap_heatmap.png',dpi=170); plt.close(fig)
    # 4 category ranking
    z=det.merge(det[['function','category']].drop_duplicates(),on='function',suffixes=('','_meta'))
    z['rank']=z.groupby(['function','dimension'])['median_gap'].rank(ascending=True,method='average')
    cat=z.groupby(['category','method'])['rank'].mean().reset_index(); cats=list(cat.category.drop_duplicates())
    pv=cat.pivot(index='category',columns='method',values='rank')
    fig,ax=plt.subplots(figsize=(13,8),constrained_layout=True)
    im=ax.imshow(pv.to_numpy(),aspect='auto',cmap='viridis_r',vmin=1,vmax=len(METHODS))
    ax.set_xticks(range(len(pv.columns)),pv.columns,rotation=55,ha='right'); ax.set_yticks(range(len(pv.index)),pv.index,fontsize=8)
    ax.set_title('Mean rank by landscape family (lower is better)'); fig.colorbar(im,ax=ax,label='mean rank')
    fig.savefig(PLOTS/'04_category_rank_heatmap.png',dpi=160); plt.close(fig)
    # 5 Pareto runtime vs gap
    fig,ax=plt.subplots(figsize=(9,7),constrained_layout=True)
    for _,r in overall.iterrows(): ax.scatter(r['median_seconds'],max(r['mean_success_1e3'],0.01),s=80,label=r['method'])
    ax.set_xscale('log'); ax.set_xlabel('Median wall-clock seconds per run (log scale)'); ax.set_ylabel('Mean success rate (%)'); ax.set_title('Runtime vs success trade-off')
    ax.legend(fontsize=7,loc='best'); fig.savefig(PLOTS/'05_runtime_success.png',dpi=160); plt.close(fig)
    # 6 paired wins/losses heatmap versus ABG
    if tests:
        t=pd.DataFrame(tests); t=t[t.noise_level==0]
        wins=t.groupby('competitor').agg(wins=('abg_wins','sum'),losses=('abg_losses','sum'),ties=('ties','sum'))
        wins['net wins']=wins['wins']-wins['losses']
        fig,ax=plt.subplots(figsize=(9,5),constrained_layout=True)
        vals=wins[['wins','losses','ties']].to_numpy().T
        im=ax.imshow(vals,aspect='auto',cmap='Blues')
        ax.set_xticks(range(len(wins.index)),wins.index,rotation=45,ha='right'); ax.set_yticks(range(3),['ABG wins','ABG losses','ties'])
        for i in range(vals.shape[0]):
            for j in range(vals.shape[1]): ax.text(j,i,str(int(vals[i,j])),ha='center',va='center',fontsize=8)
        ax.set_title('Paired trial outcomes: ABG-v30 vs competitors (all deterministic tasks)')
        fig.colorbar(im,ax=ax,label='number of paired trials'); fig.savefig(PLOTS/'06_paired_wins_losses.png',dpi=160); plt.close(fig)
    # 7 performance profile: fraction tasks within tau times the best gap
    det=summ[summ.noise_level==0].copy(); taskcols=['function','dimension']
    best=det.groupby(taskcols)['median_gap'].min().rename('best').reset_index(); det=det.merge(best,on=taskcols)
    det['ratio']=np.maximum(det['median_gap'],1e-15)/np.maximum(det['best'],1e-15)
    taus=np.geomspace(1,1000,100)
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    for m,g in det.groupby('method'):
        ratios=g['ratio'].to_numpy(float); prof=[np.mean(ratios<=t) for t in taus]; ax.plot(taus,prof,label=m)
    ax.set_xscale('log'); ax.set_xlabel('τ: ratio to best median gap on each task'); ax.set_ylabel('Fraction of tasks with ratio ≤ τ'); ax.set_ylim(-.02,1.02); ax.set_title('Performance profile across deterministic tasks'); ax.legend(fontsize=7)
    fig.savefig(PLOTS/'07_performance_profile.png',dpi=160); plt.close(fig)
    # 8+ convergence curves, median trace across trials, five representative test tasks
    trace_file=OUT/'convergence_traces.csv'
    if trace_file.exists() and trace_file.stat().st_size>0:
        tr=pd.read_csv(trace_file)
        representative=[('Sphere (shifted convex)',2),('Rosenbrock',2),('Rastrigin',2),('Schwefel',10),('Rotated Ellipsoid (convex)',30)]
        for i,(fname,d) in enumerate(representative,8):
            part=tr[(tr.function==fname)&(tr.dimension==d)]
            if part.empty: continue
            fig,ax=plt.subplots(figsize=(9,5.6),constrained_layout=True)
            for m,g in part.groupby('method'):
                gg=g.groupby('eval_index')['gap_to_best_so_far'].median()
                ax.plot(gg.index,np.maximum(gg.values,1e-15),label=m,linewidth=1.6)
            ax.set_yscale('log'); ax.set_xlabel('Real objective evaluations'); ax.set_ylabel('Median best-so-far normalized gap')
            ax.set_title(f'Convergence trace: {fname}, d={d}, budget={BUDGET}'); ax.grid(True,alpha=.25); ax.legend(fontsize=7)
            fig.savefig(PLOTS/f'{i:02d}_convergence_{i-7}.png',dpi=160); plt.close(fig)
    # 13 noise performance
    noisy=summ[summ.noise_level>0]
    if not noisy.empty:
        fig,ax=plt.subplots(figsize=(12,6),constrained_layout=True)
        pp=noisy.groupby(['method','noise_level'])['success_1e-3_pct'].mean().unstack()
        pp.plot(kind='bar',ax=ax); ax.set_ylabel('Mean success rate (%)'); ax.set_title('Noise robustness: selected 10D functions; σ relative to reference scale'); ax.legend(title='relative noise')
        fig.savefig(PLOTS/'13_noise_robustness.png',dpi=160); plt.close(fig)
    # 14 budget utilization
    fig,ax=plt.subplots(figsize=(10,5.5),constrained_layout=True)
    util=df[df.noise_level==0].groupby('method')['budget_used_pct'].median().sort_values()
    ax.barh(util.index,util.values); ax.set_xlim(0,105); ax.set_xlabel('Median fraction of objective-evaluation budget used (%)'); ax.set_title('Budget utilization (some solvers may terminate early)')
    fig.savefig(PLOTS/'14_budget_utilization.png',dpi=160); plt.close(fig)
    # 15 all-method success vs gap scatter
    fig,ax=plt.subplots(figsize=(9,6),constrained_layout=True)
    for _,r in overall.iterrows(): ax.scatter(r['mean_success_1e3'],r['mean_rank'],s=70,label=r['method'])
    ax.set_xlabel('Mean success rate (%)'); ax.set_ylabel('Mean rank (lower is better)'); ax.set_title('Quality trade-off: rank and threshold success'); ax.invert_yaxis(); ax.legend(fontsize=7)
    fig.savefig(PLOTS/'15_quality_tradeoff.png',dpi=160); plt.close(fig)


def write_report(df,summ,overall,tests):
    det=summ[summ.noise_level==0]
    best=overall.iloc[0]
    nfunc=det['function'].nunique(); ntasks=det[['function','dimension']].drop_duplicates().shape[0]
    # Aggregate significance counts after global Holm adjustment, excluding noise runs for headline.
    testdf=pd.DataFrame(tests) if tests else pd.DataFrame()
    if not testdf.empty:
        td=testdf[testdf.noise_level==0]
        sig=td.groupby('competitor')['significant_holm_0_05'].sum().to_dict()
    else: sig={}
    lines=[]
    lines += ['# ABG Top-8 Benchmark — final report','',
      '## Headline',
      f'- **Function families:** {len(ALL_NAMES)} total; **{nfunc}** have deterministic scalable or special-function benchmark rows.',
      f'- **Deterministic function × dimension tasks:** {ntasks}.',
      f'- **Algorithms:** {len(METHODS)}; **trials per task:** {TRIALS}; **objective-evaluation budget:** {BUDGET} per run.',
      f'- **Noisy stress tests:** {len(NOISE_FUNCTIONS)} selected families × relative noise levels {NOISE_LEVELS} × {NOISE_TRIALS} trials, d=10.',
      f'- **Best mean-rank method in this run:** {best["method"]} (mean rank {best["mean_rank"]:.3f}); this is a result for this exact suite, not a universal superiority claim.',
      '', '## What the plots mean',
      '- Each landscape image has two panels: a 2D domain slice with coordinates 3..d fixed at zero, and a slice through the known global minimizer with the remaining coordinates fixed at their optimal values. The color uses log10 of a dimensionless gap for readability; visual clipping never enters benchmark scores.',
      '- Normalized gap is max(0, f(x_best)-f*) divided by a method-independent reference scale estimated from 256 Sobol points before optimization. The same scale is used for all methods on that function instance.',
      '- Each algorithm has the same hard cap of 300 counted objective evaluations. Deterministic exact duplicate points use the ABG harness cache for every method; noisy evaluations are not cached. Post-run clean scoring and the reference design are audit/scoring operations and are excluded from the optimizer budget for all algorithms.',
      '- All methods are bounded to normalized coordinates [-1,1]^d. DE, Dual Annealing, DIRECT, CMA-ES, local-search baselines, Adam, and ABG have different native internal operations; fairness is defined by the same objective-evaluation cap, not equal iterations or wall-clock.',
      '- Adam uses the transparent fixed learning rate 0.03; it is not labelled tuned. Other default algorithm settings are disclosed in the script.',
      '- Paired tests compare ABG with every competitor on the same trial instances. P-values are Holm-adjusted across all tested function/dimension/noise comparisons; significance is not inferred from average ranks alone.',
      '', '## Overall ranking (deterministic suite)',
      '| Rank | Algorithm | Mean rank | Median rank | Mean success ≤1e-3 (%) | Mean success ≤1e-6 (%) | Median seconds/run | Tasks |',
      '|---:|---|---:|---:|---:|---:|---:|---:|']
    for ix,r in enumerate(overall.itertuples(index=False),1):
        lines.append(f'| {ix} | {r.method} | {r.mean_rank:.3f} | {r.median_rank:.2f} | {r.mean_success_1e3:.1f} | {r.mean_success_1e6:.1f} | {r.median_seconds:.4f} | {int(r.n_tasks)} |')
    lines += ['', '## ABG paired-test summary after global Holm correction', '| Competitor | ABG wins | ABG losses | ties | Significant task-level tests |', '|---|---:|---:|---:|---:|']
    if not testdf.empty:
        td=testdf[testdf.noise_level==0]
        for m in [m for m in METHODS if m!='ABG-v30']:
            g=td[td.competitor==m]
            lines.append(f'| {m} | {int(g.abg_wins.sum())} | {int(g.abg_losses.sum())} | {int(g.ties.sum())} | {int(g.significant_holm_0_05.sum())} |')
    lines += ['', '## Algorithm roster', '| Algorithm | Role / method |', '|---|---|',
      '| ABG-v30 | User-designed adaptive box subdivision with optional quadratic surrogate and local refinement. |',
      '| DIRECT | Deterministic partitioning of bounded domain. |',
      '| CMA-ES (compact implementation) | Covariance-adaptation evolutionary search, implemented in the bundled harness. |',
      '| Differential Evolution | Population-based differential mutation/recombination. |',
      '| Dual Annealing | Stochastic global search with generalized annealing and optional local search. |',
      '| Sobol+L-BFGS-B | Low-discrepancy start design plus bounded local refinements. |',
      '| Powell multi-start | Derivative-free local search from diverse Sobol starts. |',
      '| Adam (finite-difference, lr=0.03) | Adam-style update using finite-difference gradient estimates; fixed LR. |',
      '', '## Function families and plot catalog', '| # | Function | Category | Benchmark settings |', '|---:|---|---|---|']
    catalog=pd.read_csv(OUT/'landscape_catalog.csv') if (OUT/'landscape_catalog.csv').exists() else pd.DataFrame()
    if not catalog.empty:
        for i,r in catalog.iterrows():
            lines.append(f'| {i+1} | {r["function"]} | {r["category"]} | [Landscape](plots/landscapes/{Path(r["file"]).name}) |')
    lines += ['', '## Limitations and interpretation',
      '- Ten stochastic trials per deterministic task provide paired evidence, but this is still a finite benchmark rather than a universal proof. Several functions are classic analytic functions rather than the full official COCO/BBOB implementation with all prescribed transformations and noise models.',
      '- For some special functions, the global-best reference values are literature best-known values, and the plotted 2D slice can be harder or easier than the full-dimensional search.',
      '- Wall-clock timings are environment-dependent. The ranking is based on median dimensionless optimality gap, mean task rank, and threshold success; runtime is reported as a separate efficiency measure.',
      '- Failure rows are retained with a worst-case gap penalty, not silently dropped.',
      '', '## References',
      '- COCO BBOB test-suite overview and function families: https://coco-platform.org/testsuites/bbob/overview.html',
      '- SciPy Differential Evolution documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.differential_evolution.html',
      '- SciPy Dual Annealing documentation: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.dual_annealing.html',
      '', '## Files',
      '- `raw_results.csv`: run-level outcomes and budget accounting.',
      '- `summary_by_function_dimension.csv`: per-task medians, IQRs, success rates, runtime, failures.',
      '- `overall_ranking.csv`: aggregate ranking and coverage.',
      '- `paired_tests_vs_ABG.csv`: paired wins/losses and Holm-adjusted significance tests.',
      '- `convergence_traces.csv`: best-so-far trajectories on five representative tasks.',
      '- `plots/landscapes/`: 33 two-panel landscape/slice charts.',
      '- `plots/`: summary charts and convergence profiles.',
      '- `benchmark_top8_20plus.py`: executable, reproducible benchmark source.']
    (OUT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    # Summary text as standalone quick-read.
    quick=['ABG TOP-8 BENCHMARK — QUICK SUMMARY','',f'Algorithms: {len(METHODS)}',f'Function families: {len(ALL_NAMES)}',f'Deterministic tasks: {ntasks}',f'Trials/task: {TRIALS}',f'Budget/run: {BUDGET}',f'Noisy cases: {len(NOISE_FUNCTIONS)} families, sigma={NOISE_LEVELS}, trials={NOISE_TRIALS}', '', 'Overall deterministic ranking:']
    for i,r in enumerate(overall.itertuples(index=False),1): quick.append(f'{i:>2}. {r.method}: mean rank={r.mean_rank:.3f}, success@1e-3={r.mean_success_1e3:.1f}%, median sec={r.median_seconds:.4f}')
    quick += ['', 'Treat ranking as evidence for this suite only; do not extrapolate to universal superiority.']
    (OUT/'summary.txt').write_text('\n'.join(quick),encoding='utf-8')


def validate_outputs():
    df=pd.read_csv(OUT/'raw_results.csv')
    assert len(df)>0
    assert set(df.method.unique()).issubset(set(METHODS))
    assert (df.evals<=df.budget).all(), 'budget violation found in output'
    assert df[['function','dimension','budget','noise_level','trial','method']].duplicated().sum()==0, 'duplicate result keys found'
    assert len(pd.read_csv(OUT/'landscape_catalog.csv'))>=18
    # all method coverage should be common on every task and trial
    expected=set(METHODS)
    for key,g in df[df.noise_level==0].groupby(['function','dimension','trial']):
        if set(g.method)!=expected: raise AssertionError(f'incomplete method coverage for {key}: {set(g.method)}')
    return df


def main():
    import argparse
    parser=argparse.ArgumentParser(description='Resumable top-8 optimizer benchmark')
    parser.add_argument('--max-runs',type=int,default=None,help='maximum number of new algorithm runs in this invocation; safe to resume')
    parser.add_argument('--finalize-only',action='store_true',help='skip optimization and only build reports from existing raw results')
    parser.add_argument('--skip-landscapes',action='store_true',help='reuse existing per-function plots and only refresh tables/report/comparison charts')
    parser.add_argument('--landscapes-only',action='store_true',help='regenerate the per-function landscape figures and exit')
    args=parser.parse_args()
    t0=time.time()
    if args.landscapes_only:
        save_landscape_charts()
        return 0
    if args.finalize_only:
        complete=True
    else:
        print(f'Start: {len(METHODS)} methods; {len(get_specs())} deterministic function-dimension specs; {TRIALS} trials; budget {BUDGET}.',flush=True)
        complete=run_benchmark(args.max_runs)
    if not complete:
        print('This pass stopped at its run quota. Checkpoint saved; rerun the same command to resume.',flush=True)
        return 0
    if not args.skip_landscapes:
        save_landscape_charts()
    df,summ,overall,tests=make_summaries()
    make_comparison_plots(df,summ,overall,tests)
    write_report(df,summ,overall,tests)
    valid=validate_outputs()
    metadata={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'pandas':pd.__version__,'methods':METHODS,'budget':BUDGET,'trials':TRIALS,'dimensions':DIMS,'function_families':ALL_NAMES,'deterministic_specs':len(get_specs()),'noise_functions':NOISE_FUNCTIONS,'noise_levels':NOISE_LEVELS,'noise_trials':NOISE_TRIALS,'raw_rows':len(valid),'budget_violations':int((valid.evals>valid.budget).sum()),'duplicate_keys':int(valid[['function','dimension','budget','noise_level','trial','method']].duplicated().sum()),'elapsed_seconds_this_finalization':time.time()-t0}
    (OUT/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print('\n'.join((OUT/'summary.txt').read_text(encoding='utf-8').splitlines()[:20]),flush=True)
    print(f'Complete. Output directory: {OUT}. Finalization elapsed {metadata["elapsed_seconds_this_finalization"]:.1f}s',flush=True)

if __name__=='__main__': main()
