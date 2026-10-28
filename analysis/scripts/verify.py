#!/usr/bin/env python3
"""Independent arithmetic and frozen-package verification; no scientific model use."""
# Repository copy of Phase3B/scripts/verify.py: repo-relative input root; absent provenance-only inputs are skipped.
from pathlib import Path
import argparse,csv,json,hashlib,statistics,sys,importlib.metadata
import numpy as np
from sklearn import metrics as skm
P=Path(__file__).resolve().parents[1]
a=argparse.ArgumentParser();a.add_argument('--replay-dir',type=Path);a.add_argument('--repo',type=Path);args=a.parse_args()
cfg=json.loads((P/'config/analysis_config.json').read_text());R=args.repo or P/cfg['repo_default']
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={};rows=list(csv.DictReader((P/'tables/baseline_metrics.csv').open()));arrays={}
for scope,file in [('fold1','preds_mc25_b_fold1.npz'),('fold2','preds_mc25_a_fold2.npz')]:
 with np.load(R/'code/pipeline/cache'/file,allow_pickle=False) as z:arrays[scope]=(z['y_true'],z['y_pred'])
arrays['pooled']=tuple(np.concatenate([arrays['fold1'][i],arrays['fold2'][i]]) for i in range(2))
for scope,(y,p) in arrays.items():
 row=next(x for x in rows if x['scope']==scope)
 expected={'accuracy':skm.accuracy_score(y,p),'macro_f1':skm.f1_score(y,p,labels=range(25),average='macro',zero_division=0),'macro_precision':skm.precision_score(y,p,labels=range(25),average='macro',zero_division=0),'macro_recall':skm.recall_score(y,p,labels=range(25),average='macro',zero_division=0),'weighted_f1':skm.f1_score(y,p,labels=range(25),average='weighted',zero_division=0),'cohen_kappa':skm.cohen_kappa_score(y,p,labels=range(25)),'multiclass_mcc':skm.matthews_corrcoef(y,p)}
 for k,v in expected.items():assert abs(float(row[k])-float(v))<1e-12,(scope,k)
 with (P/f'tables/baseline_confusion_{scope}.csv').open() as f:cmrows=list(csv.DictReader(f))
 observed=np.array([[int(r[f'pred_{i:02d}']) for i in range(25)] for r in cmrows])
 assert np.array_equal(observed,skm.confusion_matrix(y,p,labels=range(25)))
checks['baseline_independent_sklearn_checks']='all3 cohorts accuracy/macro metrics/weightedF1/kappa/MCC and complete matrices agree to1e-12'
source=list(csv.DictReader((R/'code/pipeline/cache/irof_aoc_per_grain.csv').open()))
lookup={(r['sec_patch'],r['branch'],r['method']):float(r['aoc']) for r in source};gs=sorted({r['sec_patch'] for r in source})
dest=list(csv.DictReader((P/'tables/irof_all_paired_contrasts.csv').open()));maxerr=0.
for row in dest:
 vals={}
 for g in gs:
  m=row['method'];b=row['branch'];c=row['comparator'];t=row['kind']
  if t=='method_minus_random':v=lookup[g,b,m]-lookup[g,b,'random']
  elif t=='random_adjusted_branch_interaction':v=(lookup[g,'XPL',m]-lookup[g,'XPL','random'])-(lookup[g,'PPL',m]-lookup[g,'PPL','random'])
  elif t=='paired_method_comparison':v=lookup[g,b,m]-lookup[g,b,c]
  else:v=lookup[g,'XPL',m]-lookup[g,'PPL',m]
  vals.setdefault('/'.join(g.split('/')[:2]),[]).append(v)
 expected=statistics.mean(statistics.mean(v) for v in vals.values());err=abs(expected-float(row['section_equal_mean']));maxerr=max(maxerr,err);assert err<1e-12
 assert float(row['simultaneous95_all41_low'])<=float(row['section_equal_mean'])<=float(row['simultaneous95_all41_high'])
checks['all41_contrast_means_independent_stdlib_max_abs_error']=maxerr
checks['simultaneous_intervals_excluding_zero']=sum(float(r['simultaneous95_all41_low'])>0 or float(r['simultaneous95_all41_high'])<0 for r in dest)
assert checks['simultaneous_intervals_excluding_zero']==0
if args.replay_dir:
 rep=args.replay_dir;compared=[]
 for d in ['tables','figure_source_data','results','figures']:
  for p in sorted((rep/d).glob('*')):
   if not p.is_file():continue
   q=P/d/p.name;assert q.exists(),q
   if p.suffix=='.npz':
    with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
     assert set(x.files)==set(y.files)
     for k in x.files:assert np.array_equal(x[k],y[k]),(p,k)
   else:assert sha(p)==sha(q),f'Replay differs: {q}'
   compared.append(str(q.relative_to(P)))
 checks['fresh_complete_replay_files_compared']=len(compared);checks['replay_files']=compared
checks['original_inputs_unchanged']=all(sha(R/rel)==e['sha256'] for rel,e in cfg['inputs'].items() if (R/rel).exists() or not e.get('provenance_only'));assert checks['original_inputs_unchanged']
checks['verification_environment']={'python':sys.version,'numpy':np.__version__,'scikit-learn':importlib.metadata.version('scikit-learn'),'note':'sklearn used only as independent metric verifier, not by analysis/plot builder'}
(P/'results/verification.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps({k:v for k,v in checks.items() if k not in ['replay_files','verification_environment']}))
