#!/usr/bin/env python3
"""Reanalyse frozen saved records only. Never imports a model or loads a checkpoint."""
# Repository copy of publication-development/Phase3B/scripts/analyze.py. Changes: repo-relative default input root,
# provenance-only inputs may be absent, and recorded paths are relative. Scientific computations are unchanged.
from pathlib import Path
import argparse, csv, json, hashlib, itertools, os, sys, platform, importlib.metadata
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
CFG = json.loads((PACKAGE/'config/analysis_config.json').read_text())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', type=Path, default=PACKAGE/CFG['repo_default'])
parser.add_argument('--output-dir', type=Path, default=PACKAGE)
args = parser.parse_args(); ROOT=args.repo.resolve(); OUT=args.output_dir.resolve()
for d in ['tables','results','figures','figure_source_data']:(OUT/d).mkdir(parents=True,exist_ok=True)
os.environ['MPLCONFIGDIR']=str(PACKAGE/'.runtime/matplotlib')
os.environ['MPLBACKEND']='Agg'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','svg.hashsalt':'THOM-Phase3B-20261024','savefig.dpi':320})

def jwrite(rel,data):
 (OUT/rel).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def cwrite(rel,rows,fields=None):
 rows=list(rows)
 if fields is None:fields=list(rows[0]) if rows else []
 with (OUT/rel).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def readcsv(rel):
 with (ROOT/rel).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def div(a,b):return np.divide(a,b,out=np.zeros_like(np.asarray(a,dtype=float)),where=np.asarray(b)!=0)
def section(g):
 parts=g.split('/');assert len(parts)==4 and parts[-1]=='ppl-0.png',g
 return '/'.join(parts[:2])
def savefig(fig,name):
 fig.savefig(OUT/'figures'/f'{name}.png',dpi=320,facecolor='white')
 fig.savefig(OUT/'figures'/f'{name}.pdf',metadata={'CreationDate':None,'ModDate':None},facecolor='white')
 fig.savefig(OUT/'figures'/f'{name}.svg',metadata={'Date':None},facecolor='white')
 plt.close(fig)

# Hash checks precede scientific calculations. These external records are never changed.
manifest=[]
for rel,expected in CFG['inputs'].items():
 p=ROOT/rel
 if expected.get('provenance_only') and not p.exists():
  manifest.append({'relative_path':rel,'present':False,'expected_sha256':expected['sha256'],'bytes':expected['bytes'],'note':'provenance-only input; not distributed'});continue
 actual=sha(p)
 assert actual==expected['sha256'] and p.stat().st_size==expected['bytes'],f'INPUT CHANGED: {p}'
 info={'relative_path':rel,'path':os.path.relpath(p,PACKAGE),'bytes':p.stat().st_size,'sha256':actual}
 if p.suffix=='.npz':
  with np.load(p,allow_pickle=False) as z:
   info['schema']={k:{'shape':list(z[k].shape),'dtype':str(z[k].dtype)} for k in z.files}
 elif p.suffix=='.csv':
  with p.open() as f:r=csv.DictReader(f);rr=list(r);info['schema']={'columns':r.fieldnames,'rows':len(rr)}
 elif p.suffix in ['.json','.ipynb']:
  d=json.loads(p.read_text());info['schema']={'top_level_keys':list(d)}
 else:info['schema']={'kind':'source text'}
 manifest.append(info)
jwrite('results/input_manifest.json',{'inputs':manifest,'original_records_copied':False,'recorded_class_labels_source':'modality_reliance_25class.csv class/name; not independent raw annotation verification','linkage_limits':{'baseline':'nominal fold/checkpoint from filenames and run record; no grain/section rows','ablation':'class-level aggregates only; no706-row linkage','irof':'exact sec_patch + name + branch + method in same CSV; section key parsed from batch/section prefix','other_recovered_ID_sets':'not joined to outcomes without demonstrated row linkage'}})

# Baseline: exact confusion-matrix arithmetic, fixed 25-category domain.
mod=readcsv('code/pipeline/cache/modality_reliance_25class.csv')
assert len(mod)==25 and {int(r['class']) for r in mod}==set(range(25))
names={int(r['class']):r['name'] for r in mod}
class_map=[{'class_id':i,'class_name':names[i],'mapping_source':'recorded modality summary; code-level mapping audited previously'} for i in range(25)]
cwrite('tables/class_mapping.csv',class_map)
base={}
for fold,checkpoint,filename in [('fold1','B','preds_mc25_b_fold1.npz'),('fold2','A','preds_mc25_a_fold2.npz')]:
 with np.load(ROOT/'code/pipeline/cache'/filename,allow_pickle=False) as z:
  assert set(z.files)=={'y_true','y_pred'}
  y=z['y_true'];p=z['y_pred'];assert y.shape==p.shape and y.ndim==1
  assert y.dtype.kind in 'iu' and p.dtype.kind in 'iu' and y.min()>=0 and y.max()<25 and p.min()>=0 and p.max()<25
  cm=np.bincount(y*25+p,minlength=625).reshape(25,25)
  base[fold]={'cm':cm,'n':len(y),'checkpoint':checkpoint}
base['pooled']={'cm':base['fold1']['cm']+base['fold2']['cm'],'n':base['fold1']['n']+base['fold2']['n'],'checkpoint':'A+B'}
assert base['pooled']['n']==105802
metrics=[];classmetrics=[];cms=[]
for scope,d in base.items():
 cm=d['cm'];n=int(cm.sum());t=cm.sum(1);p=cm.sum(0);tp=np.diag(cm);fn=t-tp;fp=p-tp;tn=n-tp-fn-fp
 pr=div(tp,p);re=div(tp,t);f1=div(2*pr*re,pr+re);spec=div(tn,tn+fp);npv=div(tn,tn+fn)
 acc=float(tp.sum()/n);expected=float((t.astype(float)*p).sum()/n**2)
 denom=float((n*n-(p.astype(float)**2).sum())*(n*n-(t.astype(float)**2).sum()))
 mcc=float((int(tp.sum())*n-(t.astype(float)*p).sum())/np.sqrt(denom)) if denom>0 else 0.
 row={'scope':scope,'nominal_checkpoint':d['checkpoint'],'n':n,'correct':int(tp.sum()),'accuracy':acc,'macro_precision':float(pr.mean()),'macro_recall':float(re.mean()),'macro_f1':float(f1.mean()),'weighted_precision':float((pr*t).sum()/n),'weighted_recall':float((re*t).sum()/n),'weighted_f1':float((f1*t).sum()/n),'micro_precision':acc,'micro_recall':acc,'micro_f1':acc,'balanced_accuracy_present_classes':float(re[t>0].mean()),'cohen_kappa':float((acc-expected)/(1-expected)),'multiclass_mcc':mcc,'classes_with_true_support':int((t>0).sum()),'classes_with_predictions':int((p>0).sum())}
 metrics.append(row)
 for i in range(25):
  classmetrics.append({'scope':scope,'class_id':i,'class_name':names[i],'support':int(t[i]),'predicted_support':int(p[i]),'tp':int(tp[i]),'fp':int(fp[i]),'fn':int(fn[i]),'tn':int(tn[i]),'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i]),'specificity_ovr':float(spec[i]),'negative_predictive_value_ovr':float(npv[i]),'true_support_present':bool(t[i]),'predicted_support_present':bool(p[i])})
  for k in range(25):cms.append({'scope':scope,'true_class_id':i,'true_class_name':names[i],'predicted_class_id':k,'predicted_class_name':names[k],'count':int(cm[i,k]),'fraction_of_true_class':float(cm[i,k]/t[i]) if t[i] else 0.})
 cwrite(f'tables/baseline_confusion_{scope}.csv',[{'true_class_id':i,**{f'pred_{k:02d}':int(cm[i,k]) for k in range(25)}} for i in range(25)])
meanrow={'scope':'arithmetic_mean_of_two_fold_metrics','nominal_checkpoint':'not seed replication','n':None,'correct':None}
for k in metrics[0]:
 if k not in meanrow:meanrow[k]=float(np.mean([r[k] for r in metrics if r['scope'] in ['fold1','fold2']]))
metrics.append(meanrow)
cwrite('tables/baseline_metrics.csv',metrics);cwrite('tables/baseline_per_class_metrics.csv',classmetrics);cwrite('tables/baseline_confusion_long.csv',cms)
cwrite('figure_source_data/F1_confusion.csv',[r for r in cms if r['scope']=='pooled'])
cwrite('figure_source_data/F1_class_performance.csv',classmetrics)
assert abs(metrics[2]['accuracy']-0.622965539403792)<1e-14
assert abs(metrics[2]['macro_f1']-0.4285765197940208)<1e-14
historical=json.loads((ROOT/'code/pipeline/runs/2026-08-20_baseline-ranking-metrics.json').read_text())
for r in metrics[:3]:
 h=historical['pooled'] if r['scope']=='pooled' else historical['per_fold'][r['scope']]
 for k in ['accuracy','macro_f1']:assert abs(r[k]-h[k])<1e-12
jwrite('results/historical_probability_metrics.json',{'status':'RECORDED ONLY; no probability arrays reanalysed or values reconstructed from hard labels','source':os.path.relpath(ROOT/'code/pipeline/runs/2026-08-20_baseline-ranking-metrics.json',PACKAGE),'reported':{scope:historical[scope] for scope in ['per_fold','fold_mean','pooled']},'recommendation':'Exclude probability-based ranking metrics from main claims unless their aligned probability inputs become available.'})

# Ablation: paired class means only; never synthesize per-grain data.
modout=[]
for r in sorted(mod,key=lambda x:int(x['class'])):
 a=float(r['drop_noPPL']);b=float(r['drop_noXPL']);n=int(r['n']);assert n>0 and np.isfinite([a,b]).all()
 delta=b-a;assert abs(delta-float(r['xpl_reliance']))<1e-12
 modout.append({'class_id':int(r['class']),'class_name':r['name'],'n_selected_correct_grains':n,'mean_drop_PPL_filled':a,'mean_drop_XPL_filled':b,'paired_difference_of_class_means_XPL_minus_PPL':delta,'recorded_lean':r['leans'],'uncertainty':'unavailable: grain drops/sections absent'})
assert sum(r['n_selected_correct_grains'] for r in modout)==706
assert sum(r['paired_difference_of_class_means_XPL_minus_PPL']>0 for r in modout)==20
cwrite('tables/modality_class_summary.csv',modout);cwrite('figure_source_data/F2_modality.csv',modout)
w=np.array([r['n_selected_correct_grains'] for r in modout],float)
ma=np.array([r['mean_drop_PPL_filled'] for r in modout]);mb=np.array([r['mean_drop_XPL_filled'] for r in modout])
modsummary={'n':706,'classes':25,'larger_XPL_mean_drop_classes':20,'larger_PPL_mean_drop_classes':5,'class_n_range':[int(w.min()),int(w.max())],'equal_class_mean_PPL_drop':float(ma.mean()),'equal_class_mean_XPL_drop':float(mb.mean()),'equal_class_mean_difference':float((mb-ma).mean()),'n_weighted_mean_PPL_drop':float(np.average(ma,weights=w)),'n_weighted_mean_XPL_drop':float(np.average(mb,weights=w)),'n_weighted_mean_difference':float(np.average(mb-ma,weights=w)),'class_difference_range':[float((mb-ma).min()),float((mb-ma).max())],'grain_or_section_uncertainty_identifiable':False,'weighting_note':'n-weighted averages of retained class means; no missing row outcomes inferred','scope':'706 selected correct fold2 cases, historic mean-fill; fill retains channel averages and changes spatial structure/contrast/cross-modal consistency'}
jwrite('results/modality_summary.json',modsummary)

# IROF: exact complete paired record, no maps, trajectories or null draws invented.
irof=readcsv('code/pipeline/cache/irof_aoc_per_grain.csv');M=CFG['methods'];BR=CFG['branches'];allmethods=M+['random'];values={};gnames={}
for r in irof:
 g=r['sec_patch'];section(g);key=(g,r['branch'],r['method']);assert key not in values
 assert r['branch'] in BR and r['method'] in allmethods
 values[key]=float(r['aoc']);assert np.isfinite(values[key])
 if g in gnames:assert gnames[g]==r['name']
 gnames[g]=r['name']
grains=sorted(gnames,key=lambda g:(gnames[g],g));n=len(grains);assert n==30 and len(values)==360
for g,b,m in itertools.product(grains,BR,allmethods):assert (g,b,m) in values
secs=sorted({section(g) for g in grains});assert len(secs)==29
raw=[];grain_index=[]
for g in grains:
 grain_index.append({'sec_patch':g,'section_key':section(g),'batch':g.split('/')[0],'class_name':gnames[g],'grain_id_within_section':g.split('/')[2],'source_linkage':'exact same-CSV row identifier; section prefix is path-derived','fold':'nominal fold2 from checkpointA source protocol; not verified original split row'})
 for b,m in itertools.product(BR,allmethods):
  v=values[(g,b,m)];raw.append({'sec_patch':g,'section_key':section(g),'batch':g.split('/')[0],'class_name':gnames[g],'branch':b,'method':m,'recorded_aoc':v,'recorded_random_mean_aoc':values[(g,b,'random')],'method_minus_recorded_random':v-values[(g,b,'random')]})
cwrite('tables/irof_recorded_and_derived_grain_scores.csv',raw);cwrite('tables/irof_grain_identity_index.csv',grain_index)
coverage=[]
for s in secs:
 gg=[g for g in grains if section(g)==s]
 coverage.append({'section_key':s,'n_grains':len(gg),'class_names':' | '.join(sorted({gnames[g] for g in gg})),'batch':s.split('/')[0]})
cwrite('tables/irof_section_coverage.csv',coverage)
meta=[];columns=[]
for m,b in itertools.product(M,BR):
 meta.append({'contrast_id':f'D|{m}|{b}','kind':'method_minus_random','method':m,'comparator':'recorded random mean','branch':b})
 columns.append([values[(g,b,m)]-values[(g,b,'random')] for g in grains])
for m in M:
 meta.append({'contrast_id':f'I|{m}','kind':'random_adjusted_branch_interaction','method':m,'comparator':'XPL delta minus PPL delta','branch':'XPL-PPL'})
 columns.append([(values[(g,'XPL',m)]-values[(g,'XPL','random')])-(values[(g,'PPL',m)]-values[(g,'PPL','random')]) for g in grains])
for a,b in itertools.combinations(M,2):
 for br in BR:
  meta.append({'contrast_id':f'P|{a}|{b}|{br}','kind':'paired_method_comparison','method':a,'comparator':b,'branch':br})
  columns.append([values[(g,br,a)]-values[(g,br,b)] for g in grains])
for m in allmethods:
 meta.append({'contrast_id':f'R|{m}','kind':'raw_branch_difference','method':m,'comparator':'raw XPL AOC minus raw PPL AOC','branch':'XPL-PPL'})
 columns.append([values[(g,'XPL',m)]-values[(g,'PPL',m)] for g in grains])
X=np.array(columns).T;assert X.shape==(30,41)
def grouped_mean(x,gs):
 ss=sorted({section(g) for g in gs})
 return np.array([x[[section(g)==s for g in gs]].mean(0) for s in ss]),ss
SX,sec_order=grouped_mean(X,grains);theta=SX.mean(0);se=SX.std(0,ddof=1)/np.sqrt(len(SX))
rng=np.random.default_rng(CFG['bootstrap_seed']);counts=rng.multinomial(len(SX),np.full(len(SX),1/len(SX)),size=CFG['bootstrap_draws'])
boot=counts@SX/len(SX);lo,hi=np.quantile(boot,[.025,.975],axis=0)
z=np.divide(boot-theta,se,out=np.zeros_like(boot),where=se>0);maxz=np.max(np.abs(z),axis=1);q=float(np.quantile(maxz,.95))
simlo=theta-q*se;simhi=theta+q*se
np.savez_compressed(OUT/'results/bootstrap_contrast_means.npz',contrast_means=boot,contrast_ids=np.array([m['contrast_id'] for m in meta]),section_keys=np.array(sec_order))
contrasts=[]
for k,m in enumerate(meta):
 sd=float(SX[:,k].std(ddof=1))
 contrasts.append({**m,'n_grains':30,'n_section_keys':29,'section_equal_mean':float(theta[k]),'grain_equal_mean':float(X[:,k].mean()),'section_sd':sd,'paired_standardized_section_effect_dz':float(theta[k]/sd) if sd>0 else None,'pointwise95_low':float(lo[k]),'pointwise95_high':float(hi[k]),'simultaneous95_all41_low':float(simlo[k]),'simultaneous95_all41_high':float(simhi[k]),'positive_grains':int((X[:,k]>0).sum()),'zero_grains':int((X[:,k]==0).sum()),'negative_grains':int((X[:,k]<0).sum()),'grain_median':float(np.median(X[:,k])),'grain_min':float(X[:,k].min()),'grain_max':float(X[:,k].max())})
cwrite('tables/irof_all_paired_contrasts.csv',contrasts)
cwrite('tables/irof_method_minus_random.csv',[r for r in contrasts if r['kind']=='method_minus_random'])
cwrite('tables/irof_branch_interactions.csv',[r for r in contrasts if r['kind']=='random_adjusted_branch_interaction'])
cwrite('tables/irof_paired_method_comparisons.csv',[r for r in contrasts if r['kind']=='paired_method_comparison'])
cwrite('tables/irof_raw_branch_differences.csv',[r for r in contrasts if r['kind']=='raw_branch_difference'])
cwrite('tables/irof_section_contrasts.csv',[{'section_key':s,'contrast_id':meta[k]['contrast_id'],'section_mean':float(SX[i,k])} for i,s in enumerate(sec_order) for k in range(41)])
cwrite('tables/irof_grain_contrasts.csv',[{'sec_patch':g,'section_key':section(g),'class_name':gnames[g],'contrast_id':meta[k]['contrast_id'],'grain_value':float(X[i,k])} for i,g in enumerate(grains) for k in range(41)])
influence=[]
for unit,labels in [('section',secs),('grain',grains),('class',sorted(set(gnames.values())))]:
 for label in labels:
  keep=np.array([(section(g)!=label if unit=='section' else g!=label if unit=='grain' else gnames[g]!=label) for g in grains])
  agg,ss=grouped_mean(X[keep],[g for g,kp in zip(grains,keep) if kp]);v=agg.mean(0)
  for k in range(41):influence.append({'removed_unit':unit,'removed_id':label,'contrast_id':meta[k]['contrast_id'],'remaining_grains':int(keep.sum()),'remaining_sections':len(ss),'estimate':float(v[k]),'change_from_full':float(v[k]-theta[k])})
cwrite('tables/irof_leave_one_out.csv',influence)
classes=sorted(set(gnames.values()));classrows=[];classmacro=[]
for c in classes:
 mask=np.array([gnames[g]==c for g in grains]);sxc,cs=grouped_mean(X[mask],[g for g,kp in zip(grains,mask) if kp]);ct=sxc.mean(0);classmacro.append(ct)
 for k in range(41):classrows.append({'class_name':c,'contrast_id':meta[k]['contrast_id'],'n_grains':int(mask.sum()),'n_section_keys':len(cs),'grain_equal_mean':float(X[mask,k].mean()),'section_equal_mean_within_class':float(ct[k]),'uncertainty':'descriptive only; five grains per class'})
cwrite('tables/irof_class_descriptive.csv',classrows)
robust=[]
for k,m in enumerate(meta):
 row={**m,'section_equal_mean':float(theta[k]),'grain_equal_mean':float(X[:,k].mean()),'class_macro_section_equal_mean':float(np.mean(classmacro,axis=0)[k]),'median_section_contrast':float(np.median(SX[:,k])),'median_grain_contrast':float(np.median(X[:,k]))}
 for unit in ['section','grain','class']:
  vals=[r['estimate'] for r in influence if r['contrast_id']==m['contrast_id'] and r['removed_unit']==unit]
  row[f'leave_one_{unit}_min']=min(vals);row[f'leave_one_{unit}_max']=max(vals)
 robust.append(row)
cwrite('tables/irof_aggregation_and_influence_summary.csv',robust)

# Existing rounded aggregate curves are not per-grain curves or independent runs.
curve_rows=[];runchecks=[]
for rel in CFG['inputs']:
 if rel.endswith('_irof-fidelity.json'):
  d=json.loads((ROOT/rel).read_text());diffs=[]
  for m,bb in d.get('aoc_mean',{}).items():
   for b,v in bb.items():diffs.append(abs(float(v)-np.mean([values[(g,b,m)] for g in grains])))
  rec={'source':rel,'recorded_date':d.get('date'),'n_grains':d.get('n_grains'),'config':d.get('config'),'aggregate_aoc_max_abs_rounding_difference_from_CSV':max(diffs) if diffs else None,'rounded_mean_curve_payload_sha256':hashlib.sha256(json.dumps(d.get('mean_curves',{}),sort_keys=True).encode()).hexdigest(),'has_per_grain_curves':False,'has_individual_random_orders':False}
  runchecks.append(rec)
  for key,curve in d.get('mean_curves',{}).items():
   for frac,val in zip(d['fractions'],curve):curve_rows.append({'record':rel,'method_and_branch':key,'segment_fraction_grid':frac,'recorded_rounded_mean_normalized_probability':val,'scope':'aggregate only, no per-grain variability'})
if curve_rows:cwrite('tables/historical_rounded_mean_curves.csv',curve_rows)
jwrite('results/historical_IROF_record_comparison.json',{'records':runchecks,'warning':'Dated records may duplicate the same aggregate payload. Matching rounded means does not establish independent runs or complete input/curve provenance.'})
mc={}
for lab,b in [('first_half',boot[:len(boot)//2]),('second_half',boot[len(boot)//2:])]:
 a,c=np.quantile(b,[.025,.975],axis=0);mc[lab]={'pointwise_low':a.tolist(),'pointwise_high':c.tolist()}
a=np.array(mc['first_half']['pointwise_low']);b=np.array(mc['second_half']['pointwise_low']);c=np.array(mc['first_half']['pointwise_high']);d=np.array(mc['second_half']['pointwise_high'])
mc['max_abs_half_draw_endpoint_difference']=float(max(np.max(abs(a-b)),np.max(abs(c-d))))
jwrite('results/bootstrap_diagnostics.json',{'draws':CFG['bootstrap_draws'],'seed':CFG['bootstrap_seed'],'n_clusters':29,'family_contrasts':41,'simultaneous_max_standardized_critical_value':q,'split_draw_endpoint_diagnostic':mc,'zero_standard_errors':int((se==0).sum()),'p_values_generated':False,'assumptions':CFG['inference_limits']})
summary={'baseline':metrics,'modality':modsummary,'irof':{'grains':30,'section_keys':29,'classes':{c:sum(gnames[g]==c for g in grains) for c in classes},'batch_counts':{b:sum(g.split('/')[0]==b for g in grains) for b in sorted({g.split('/')[0] for g in grains})},'raw_aoc_range':[min(values.values()),max(values.values())],'negative_raw_aoc_rows':sum(v<0 for v in values.values()),'complete_grid':True,'excluded_rows':0,'random_order_dispersion_available':False,'pixel_area_trajectory_available':False,'aggregate_rounded_curves_available':bool(curve_rows),'contrasts':contrasts,'aggregation_robustness':robust,'coverage':coverage}}
jwrite('results/analysis_summary.json',summary)
jwrite('results/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'numpy':np.__version__,'matplotlib':matplotlib.__version__,'backend':matplotlib.get_backend(),'workload':'Numeric saved-output reanalysis and CPU bootstrap/plotting only. No torch import, model, checkpoint or image inference.','dependency_installations':False})
(OUT/'results/requirements-analysis.txt').write_text(f'numpy=={np.__version__}\nmatplotlib=={matplotlib.__version__}\n')

# Figure source tables are explicit exact subsets, not values read back from images.
cwrite('figure_source_data/F3_grain_deltas.csv',[r for r in raw if r['method']!='random'])
cwrite('figure_source_data/F3_intervals.csv',[r for r in contrasts if r['kind']=='method_minus_random'])
cwrite('figure_source_data/F4_grain_deltas.csv',[r for r in raw if r['method']!='random'])
cwrite('figure_source_data/F4_branch_interactions.csv',[r for r in contrasts if r['kind']=='random_adjusted_branch_interaction'])

# F1 baseline. Explicitly no inferred independent-grain confidence intervals.
fig,axes=plt.subplots(1,2,figsize=(14,10),gridspec_kw={'width_ratios':[1.35,1]})
cm=base['pooled']['cm'];norm=div(cm,cm.sum(1)[:,None]);im=axes[0].imshow(norm,vmin=0,vmax=1,cmap='Blues',aspect='equal')
labels=[f'{i:02d}  {names[i]}' for i in range(25)]
axes[0].set(xticks=range(25),xticklabels=[f'{i:02d}' for i in range(25)],yticks=range(25),yticklabels=labels,xlabel='Recorded predicted category ID',title='A  Pooled row-normalized confusion')
axes[0].tick_params(axis='x',labelsize=7,rotation=90);axes[0].tick_params(axis='y',labelsize=8)
fig.colorbar(im,ax=axes[0],fraction=.046,pad=.03,label='Fraction within recorded true category')
y=np.arange(25)
for scope,offset,color,marker in [('fold1',-.16,'#D55E00','s'),('fold2',.16,'#0072B2','^'),('pooled',0,'#222222','o')]:
 vals=[r['f1'] for r in classmetrics if r['scope']==scope]
 axes[1].scatter(vals,y+offset,s=24,color=color,marker=marker,label=scope,zorder=3)
axes[1].set(yticks=y,yticklabels=labels,xlim=(-.02,1.02),xlabel='Per-category F1',title='B  Category heterogeneity');axes[1].invert_yaxis();axes[1].tick_params(axis='y',labelsize=8);axes[1].grid(axis='x',alpha=.18);axes[1].legend(loc='lower right',fontsize=8)
fig.suptitle('Saved baseline predictions: 105,802 rows in two nominal evaluation folds',fontsize=13,y=.985)
fig.subplots_adjust(left=.16,right=.98,bottom=.10,top=.93,wspace=.62)
fig.text(.02,.025,'Fold 1: 52,792 rows (checkpoint B); fold 2: 53,010 (A). No row-to-section IDs or probability arrays.\nRecorded class taxonomy includes non-mineral categories. Counts and source data are supplied; no image/model rerun.',fontsize=9)
savefig(fig,'F1_Baseline_Performance')

# F2 ablation. The absence of valid standard errors is made visible.
fig,axes=plt.subplots(1,2,figsize=(14,10),gridspec_kw={'width_ratios':[1.5,1]})
y=np.arange(25);a=np.array([r['mean_drop_PPL_filled'] for r in modout]);b=np.array([r['mean_drop_XPL_filled'] for r in modout]);dd=b-a
axes[0].hlines(y,np.minimum(a,b),np.maximum(a,b),color='#aaaaaa',lw=1)
axes[0].scatter(a,y,s=28,color='#0072B2',label='PPL mean-filled');axes[0].scatter(b,y,s=28,color='#D55E00',label='XPL mean-filled')
axes[0].set(yticks=y,yticklabels=[f"{r['class_id']:02d}  {r['class_name']}  (n={r['n_selected_correct_grains']})" for r in modout],xlabel='Recorded mean true-class probability drop',title='A  Paired class-level means')
axes[0].tick_params(axis='y',labelsize=8);axes[0].invert_yaxis()
fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.59,.958),ncol=2,frameon=False,fontsize=9)
axes[1].scatter(dd,y,color=['#D55E00' if v>0 else '#0072B2' for v in dd],s=30)
axes[1].axvline(0,color='#777777',lw=1);axes[1].set(yticks=y,yticklabels=[f'{i:02d}' for i in range(25)],xlabel='XPL-filled drop minus PPL-filled drop',title='B  Descriptive contrast; 20/25 positive');axes[1].invert_yaxis()
for ax in axes:ax.grid(axis='x',alpha=.18)
fig.suptitle('Modality ablation: 706 selected correctly classified fold-2 grains',fontsize=13,y=.985)
fig.subplots_adjust(left=.27,right=.98,bottom=.11,top=.89,wspace=.28)
fig.text(.02,.025,'Class means only: per-grain drops and section linkage are absent, so no confidence intervals are inferred.\nMean-fill retains each image\'s channel averages while destroying spatial information; it does not isolate interference colour.',fontsize=9)
savefig(fig,'F2_Modality_Ablation')

# F3 relative scores. Fixed method order, no ranked bars or validity stars.
show={'Grad-CAM':'Grad-CAM','Chefer branch':'Chefer branch*','Chefer cross':'Chefer cross*','cross-attn':'Cross-attention','rollout':'Rollout'}
fig,axes=plt.subplots(1,2,figsize=(13,7),sharey=True,sharex=True);plotrng=np.random.default_rng(CFG['plot_seed'])
for ax,b,color in zip(axes,BR,['#0072B2','#D55E00']):
 for i,m in enumerate(M):
  k=next(k for k,t in enumerate(meta) if t['contrast_id']==f'D|{m}|{b}')
  ax.scatter(X[:,k],i+plotrng.uniform(-.11,.11,n),s=14,color=color,alpha=.35,zorder=2)
  ax.hlines(i,simlo[k],simhi[k],color='#888888',lw=1.2,zorder=3)
  ax.hlines(i,lo[k],hi[k],color=color,lw=4,zorder=4);ax.plot(theta[k],i,'o',color='#111111',ms=5,zorder=5)
 ax.axvline(0,color='#555555',ls='--',lw=1);ax.set(yticks=range(5),yticklabels=[show[m] for m in M],xlabel='AOC(method) − recorded mean random AOC',title=b);ax.grid(axis='x',alpha=.15)
axes[0].invert_yaxis();axes[1].tick_params(labelleft=True)
fig.suptitle('Paired deletion effects: 30 correct grains / 29 recorded section groups',fontsize=13,y=.985)
fig.subplots_adjust(left=.17,right=.98,bottom=.22,top=.88,wspace=.38)
fig.text(.02,.035,'Dots: recorded grain pairs; black: equal-section mean. Thick lines: 95% pointwise section-bootstrap intervals;\nthin grey: approximate simultaneous 95% intervals across all 41 contrasts. 10,000 shared paired draws.\n*Architecture-specific adaptations. Five implemented methods include structural baselines; no universal ranking.\nOne selected XAI fold, six classes, fixed mean-fill / segment-fraction protocol; random reference averages three orders.',fontsize=9)
savefig(fig,'F3_Paired_Deletion_Effects')

# F4 raw grain variation plus the correctly paired difference-in-differences.
fig,axes=plt.subplots(1,2,figsize=(15,11),gridspec_kw={'width_ratios':[1.5,1]})
G=X[:,:10];lim=float(np.abs(G).max());im=axes[0].imshow(G,cmap='RdBu_r',vmin=-lim,vmax=lim,aspect='auto')
abbrev={'Grad-CAM':'GC','Chefer branch':'CB*','Chefer cross':'CC*','cross-attn':'CA','rollout':'RO'}
axes[0].set(xticks=range(10),xticklabels=[f'{abbrev[m]}\n{b}' for m,b in itertools.product(M,BR)],yticks=range(30),yticklabels=[g.replace('GMAS_LOTE2/','B2/').replace('GMAS_LOTE/','B1/').replace('/ppl-0.png','') for g in grains],title='A  Every recorded grain-level method-minus-random score',xlabel='Method and branch',ylabel='Recorded batch / section / grain key')
axes[0].tick_params(axis='y',labelsize=7.5);axes[0].tick_params(axis='x',labelsize=8)
fig.colorbar(im,ax=axes[0],fraction=.04,pad=.03,label='AOC difference (symmetric colour scale)')
for i,m in enumerate(M):
 k=next(k for k,t in enumerate(meta) if t['contrast_id']==f'I|{m}')
 axes[1].scatter(X[:,k],i+plotrng.uniform(-.11,.11,n),color='#999999',s=13,alpha=.4)
 axes[1].hlines(i,simlo[k],simhi[k],color='#888888',lw=1.2)
 axes[1].hlines(i,lo[k],hi[k],color='#009E73',lw=4);axes[1].plot(theta[k],i,'ko',ms=5)
axes[1].axvline(0,ls='--',color='#555555',lw=1);axes[1].set(yticks=range(5),yticklabels=[show[m] for m in M],xlabel='(Method − random) XPL − (Method − random) PPL',title='B  Paired branch interactions');axes[1].invert_yaxis();axes[1].grid(axis='x',alpha=.15)
fig.suptitle('Grain variability and branch contrasts in the retained IROF sample',fontsize=13,y=.985)
fig.subplots_adjust(left=.13,right=.98,bottom=.15,top=.93,wspace=.52)
fig.text(.02,.028,'30 selected correct grains; six categories × five grains; 29 section keys. No per-grain AOC is truncated.\nB1 = GMAS_LOTE; B2 = GMAS_LOTE2. GC = Grad-CAM; CB/CC = Chefer branch/cross adaptations; CA = cross-attention; RO = rollout.\nPanel B: black = equal-section mean; thick = pointwise 95%; grey = simultaneous 95% (41 contrasts), under section exchangeability.\nMask pixel areas, per-grain curves and individual null-order variation are unavailable. Section groups may share unknown parents.',fontsize=9)
savefig(fig,'F4_Grain_Variability_and_Branch_Contrasts')

# Recheck the original bytes after all reads/calculations.
assert all(sha(ROOT/r)==e['sha256'] for r,e in CFG['inputs'].items() if (ROOT/r).exists() or not e.get('provenance_only'))
jwrite('results/transformation_and_exclusion_log.json',{'source_hashes_unchanged':True,'baseline':'Numeric int64 arrays only; 25-label confusion arithmetic; zero-division values0 with support flags; no ranking metrics inferred','modality':'Sorted by recorded class ID; added difference of class means and n/equal-class weighted summaries; no rows synthesized','irof':'Exact source IDs, finite values and full30x2x6 grid; path-derived section groups; all rows retained; complete paired linear contrasts','uncertainty':'Common section resampling over all41 contrasts; pointwise percentiles and approximate simultaneous max-standardized intervals; no P values; no causal/confirmatory inference','influence':'Leave-one-grain, section and class; aggregation recalculated after each exclusion; these sensitivity removals do not discard full-analysis rows','figures':'Recorded numeric values only; fixed seeds for display-only y jitter; no x-value changes, image panels or model maps','unavailable':'baseline grain/section row IDs and probabilities; ablation grain drops/section keys; full IROF per-grain curves, pixel areas and null repeats; higher parent IDs'})
print(json.dumps({'status':'analysis_and_figures_complete','baseline_n':105802,'ablation_n':706,'irof_grains':30,'irof_sections':29,'paired_contrasts':41,'bootstrap_draws':CFG['bootstrap_draws'],'simultaneous_factor':q,'figures':4}))
