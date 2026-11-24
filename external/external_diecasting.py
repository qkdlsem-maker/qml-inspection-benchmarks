"""Reproduce the frozen chronological shot-level validation in two separate stages."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
from io import BytesIO
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import scipy
from scipy.linalg import hadamard
import sklearn
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits


COMMIT = "3010621441321d2d51f8043d0edc4ac6440cd80f"
BLOB = "5319bc94337c0ad0284553f2611f5ec31c13fd98"
FILE_SIZE = 17_920_698
URL = f"https://raw.githubusercontent.com/yeonduhaeyo/diecasting-project/{COMMIT}/data/train.csv"
PROTOCOL_SHA = "51bdf7844e08ef726cf3beefe358c14c380157777488688ac5611a5ec0231849"
SENSORS = ["molten_temp", "sleeve_temperature", "low_section_speed", "high_section_speed",
           "cast_pressure", "biscuit_thickness", "physical_strength",
           "upper_mold_temp1", "upper_mold_temp2", "upper_mold_temp3",
           "lower_mold_temp1", "lower_mold_temp2", "lower_mold_temp3", "Coolant_temperature",
           "EMS_operation_time", "facility_operation_cycleTime", "production_cycletime"]
MODELS = ["Q4 + RZZ","PCA4 RBF","Q4 no RZZ","PCA4 logistic","Chronology only"]
POLICIES = ["Largest remainder","Cumulative volume"]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bytes_for_analysis(file):
    if file:
        raw = Path(file).read_bytes()
    else:
        request = Request(URL, headers={"User-Agent":"QML-inspection-reproduction"})
        with urlopen(request, timeout=120) as response:
            raw = response.read()
    git_hash = hashlib.sha1(f"blob {len(raw)}\0".encode("ascii") + raw).hexdigest()
    if len(raw) != FILE_SIZE or git_hash != BLOB:
        raise ValueError("The complete source does not match its fixed byte length and Git blob.")
    return raw


def write_json(path, vals):
    path.write_text(json.dumps(vals, ensure_ascii=False, indent=2, allow_nan=False) + "\n",encoding="utf-8")


def schema(raw):
    for encoding in ("utf-8-sig","cp949"):
        try:
            raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("The CSV could not be decoded under UTF-8 or CP949.")
    columns = pd.read_csv(BytesIO(raw),encoding=encoding,nrows=0).columns.to_list()
    simple = {c.casefold():c for c in columns}
    sensors = []
    for sensor in SENSORS:
        variants = [sensor.casefold()]
        if "mold_temp" in sensor:
            variants.append(sensor.replace("mold_temp","mold_temperature").casefold())
        matches = [simple[v] for v in variants if v in simple]
        if len(matches)>1:
            raise ValueError(f"Ambiguous column for {sensor}: {matches}")
        sensors.extend(matches)
    if len(sensors)<4:
        raise ValueError("Fewer than four eligible source sensor columns.")
    return encoding,sensors,simple["registration_time"],simple["passorfail"],simple.get("mold_code")


def targets(raw, encoding, target):
    values = pd.read_csv(BytesIO(raw), encoding=encoding, usecols=[target],dtype=str)[target]
    return pd.to_numeric(values,errors="coerce")


def four_qubit_states(components, coupled):
    z = 1-2*((np.arange(16)[:,None] >> np.arange(3,-1,-1)) & 1)
    phase = components @ z.T
    if coupled:
        phase += (components[:,:-1]*components[:,1:]) @ (z[:,:-1]*z[:,1:]).T
    diagonal = np.exp(-0.5j*phase)
    states = ((diagonal/4.0) @ (hadamard(16)/4.0)) * diagonal
    np.testing.assert_allclose(np.linalg.norm(states,axis=1),1.0,atol=1e-12)
    return states


def gram(left,right):
    values = np.empty((len(left),len(right)),dtype=np.float64)
    adjoint = right.conj().T
    for start in range(0,len(left),192):
        block = slice(start,start+192)
        values[block] = np.abs(left[block] @ adjoint)**2
    return values


def fit_and_score(train_x,train_y,test_x):
    transform = make_pipeline(SimpleImputer(strategy="median"),VarianceThreshold(),StandardScaler())
    xt = transform.fit_transform(train_x)
    xv = transform.transform(test_x)
    if xt.shape[1]<4:
        raise ValueError("Fewer than four variable numeric training features.")
    pca = PCA(n_components=4,svd_solver="full")
    pc = pca.fit_transform(xt)
    pv = pca.transform(xv)
    gamma = 1/(pc.shape[1]*pc.var())
    rbf = SVC(C=1,class_weight="balanced",kernel="rbf",gamma=gamma).fit(pc,train_y)
    logistic = LogisticRegression(C=1,class_weight="balanced",solver="lbfgs",max_iter=2000,tol=1e-8).fit(pc,train_y)
    scores = {"PCA4 RBF":rbf.decision_function(pv),"PCA4 logistic":logistic.decision_function(pv)}
    scale = 0.5/np.sqrt(np.mean(pc**2))
    for name, coupled in (("Q4 + RZZ",True),("Q4 no RZZ",False)):
        training = four_qubit_states(pc*scale,coupled)
        testing = four_qubit_states(pv*scale,coupled)
        matrix = gram(training,training)
        np.testing.assert_allclose(matrix.diagonal(),1.0,atol=1e-12)
        model = SVC(C=1,kernel="precomputed",class_weight="balanced").fit(matrix,train_y)
        del matrix
        scores[name]=np.concatenate([model.decision_function(gram(testing[i:i+192],training))
                                     for i in range(0,len(testing),192)])
    scores["Chronology only"] = -np.arange(len(pv),dtype=float)
    for score in scores.values():
        if not np.isfinite(score).all():
            raise ValueError("Nonfinite decision scores.")
    retained = transform.get_feature_names_out(train_x.columns).tolist()
    return scores, {"retained_sensors":retained,"pca_variance":float(pca.explained_variance_ratio_.sum()),
                    "quantum_phase_scale":float(scale),"rbf_gamma":float(gamma)}


def quotas(sizes, policy):
    budget=(int(sizes.sum())+9)//10
    if policy=="Largest remainder":
        q,r=np.divmod(budget*sizes,int(sizes.sum()))
        q[np.argsort(-r,kind="stable")[:budget-int(q.sum())]]+=1
    else:
        q=np.diff((np.cumsum(sizes)+9)//10,prepend=0)
    assert int(q.sum())==budget and ((q>=0)&(q<=sizes)).all()
    return q


def freeze(raw, output):
    encoding,sensors,date_col,target,mold = schema(raw)
    data = pd.read_csv(BytesIO(raw),encoding=encoding,usecols=[date_col]+sensors+([mold] if mold else []),dtype=str)
    time = pd.to_datetime(data[date_col],format="mixed",errors="coerce")
    # This is solely a completeness predicate; no held-out label value enters the model or ranking.
    valid_label = targets(raw,encoding,target).isin([0,1])
    eligible = time.notna() & valid_label
    days = pd.Index(time[eligible].dt.normalize().unique()).sort_values()
    cut = 7*len(days)//10
    if min(cut,len(days)-cut)<2:
        raise ValueError("Insufficient distinct recorded dates.")
    order = time[eligible].sort_values(kind="stable").index
    train = order[time[order].dt.normalize()<days[cut]]
    test = order[time[order].dt.normalize()>=days[cut]]
    train_base_size=len(train)
    if len(train)>2048:
        smallest=set(sorted(train,key=lambda i:hashlib.sha256(str(i).encode("ascii")).digest())[:2048])
        train = pd.Index([i for i in train if i in smallest])
    ytrain = targets(raw,encoding,target).loc[train].to_numpy(dtype=np.int8)
    if set(ytrain)!={0,1}:
        raise ValueError("Hash-selected training rows do not contain both classes.")
    x = data[sensors].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    with threadpool_limits(limits=1):
        scores,model_metadata = fit_and_score(x.loc[train],ytrain,x.loc[test])
    scored = pd.DataFrame({"source_row":test.to_numpy(),"date":time[test].dt.strftime("%Y-%m-%d").to_numpy()})
    for name in MODELS:
        scored[name]=scores[name]
    # The file has no outcomes; float64 scores survive exact roundtrip in pandas.
    scored.to_csv(output/"frozen_scores.csv",index=False,float_format="%.17g",lineterminator="\n")
    group_rows=[g.index.to_numpy() for _,g in scored.groupby("date",sort=True)]
    sizes=np.array([len(r) for r in group_rows],dtype=int)
    chosen=[]
    for policy in POLICIES:
        for rows,q in zip(group_rows,quotas(sizes,policy)):
            for model in MODELS:
                ranks=np.argsort(-scored[model].to_numpy()[rows],kind="stable")
                picks=rows[ranks[:q]]
                chosen.extend((policy,model,scored.date.iloc[r],int(scored.source_row.iloc[r])) for r in picks)
    pd.DataFrame(chosen,columns=["policy","model","date","source_row"]).to_csv(output/"frozen_inspections.csv",index=False,lineterminator="\n")
    meta = {"source_sha256":digest(raw),"git_blob":BLOB,"commit":COMMIT,"protocol_sha256":PROTOCOL_SHA,
            "source_bytes":len(raw),"source_rows":len(data),"timestamp_field":date_col,"label_field":target,
            "invalid_time_rows":int(time.isna().sum()),"invalid_target_rows":int((~valid_label).sum()),
            "excluded_union":int((~eligible).sum()),"eligible_rows":int(eligible.sum()),
            "encoding":encoding,"train_start":str(days[0].date()),"train_end":str(days[cut-1].date()),
            "test_start":str(days[cut].date()),"test_end":str(days[-1].date()),"train_days":cut,"test_days":len(days)-cut,
            "eligible_train":train_base_size,"sampled_train":len(train),"sampled_train_defects":int(ytrain.sum()),
            "test_units":len(test),"inspection_slots":int((len(test)+9)//10),"input_sensors":sensors,
            **model_metadata, "software":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
            "scipy":scipy.__version__,"sklearn":sklearn.__version__},"sampled_train_ids_sha256":digest(np.array(train,dtype="<i8").tobytes()),
            "test_source_rows_sha256":digest(np.array(test,dtype="<i8").tobytes()),
            "scores_sha256":digest((output/"frozen_scores.csv").read_bytes()),
            "selections_sha256":digest((output/"frozen_inspections.csv").read_bytes())}
    if mold:
        def mold_counts(rows):
            return {str(k):int(v) for k,v in data.loc[rows,mold].fillna("(missing)").value_counts().items()}
        meta["mold_code_counts"]={"training_eligible":mold_counts(order[time[order].dt.normalize()<days[cut]]),"sampled_training":mold_counts(train),"test":mold_counts(test)}
    write_json(output/"external_freeze.json",meta)
    print("Frozen",len(test),"test scores per model over",len(days)-cut,"dates.",meta["scores_sha256"])


def evaluate(raw,output,protocol_sha=PROTOCOL_SHA):
    manifest = json.loads((output/"external_freeze.json").read_text(encoding="utf-8"))
    assert manifest["source_sha256"] == digest(raw) and manifest["protocol_sha256"] == protocol_sha
    for f,k in (("frozen_scores.csv","scores_sha256"),("frozen_inspections.csv","selections_sha256")):
        assert digest((output/f).read_bytes()) == manifest[k]
    score = pd.read_csv(output/"frozen_scores.csv",float_precision="round_trip")
    selections=pd.read_csv(output/"frozen_inspections.csv")
    y=targets(raw,manifest["encoding"],manifest["label_field"]).loc[score.source_row].to_numpy(dtype=np.int8)
    assert np.isin(y,[0,1]).all()
    day_indices=[(day,group.index.to_numpy()) for day,group in score.groupby("date",sort=True)]
    byday=[]
    for policy in POLICIES:
        for (day,rows),q in zip(day_indices,quotas(np.array([len(rows) for _,rows in day_indices]),policy)):
            for name in MODELS:
                s=score[name].to_numpy()[rows]
                order=np.argsort(-s,kind="stable")
                mask=(selections.policy==policy)&(selections.model==name)&(selections.date==day)
                assert score.source_row.to_numpy()[rows[order[:q]]].tolist()==selections.loc[mask,"source_row"].to_list()
                labels=y[rows]
                hits=int(labels[order[:q]].sum())
                tie_min=tie_max=tie_expected=0
                if q:
                    boundary=s[order[q-1]]
                    tied=s==boundary
                    above=s>boundary
                    fixed=int(labels[above].sum())
                    nt=int(tied.sum())
                    ntpos=int(labels[tied].sum())
                    free=int(q-above.sum())
                    tie_min=fixed+max(0,free-(nt-ntpos))
                    tie_max=fixed+min(free,ntpos)
                    tie_expected=fixed+free*ntpos/nt
                assert tie_min<=hits<=tie_max
                byday.append((policy,name,day,len(rows),int(q),int(labels.sum()),hits,tie_min,tie_max,tie_expected))
    daily=pd.DataFrame(byday, columns=["policy","model","date","units","slots","defects","hits","tie_min","tie_max","tie_expected"])
    daily.to_csv(output/"external_by_date.csv",index=False)
    result=[]
    for (policy,name),g in daily.groupby(["policy","model"],sort=False):
        pos=int(y.sum())
        total=int(g.hits.sum())
        result.append({"policy":policy,"model":name,"units":len(y),"defects":pos,"slots":int(g.slots.sum()),
                       "hits":total,"recall":total/pos if pos else np.nan,
                       "test_dates":len(g),"positive_dates":int((g.defects>0).sum()),"dates_reached":int((g.hits>0).sum()),
                       "tie_min":int(g.tie_min.sum()),"tie_max":int(g.tie_max.sum()),"tie_expected":g.tie_expected.sum(),
                       "pooled_ap":average_precision_score(y,score[name]) if pos else np.nan,
                       "uniform_expected":float((g.slots*g.defects/g.units).sum()),"daily_ceiling":int(np.minimum(g.slots,g.defects).sum())})
    totals=pd.DataFrame(result)
    totals.to_csv(output/"external_summary.csv",index=False)
    primary=daily.loc[daily.policy.eq("Largest remainder")]
    q4=primary.loc[primary.model.eq("Q4 + RZZ")].set_index("date").sort_index()
    rbf=primary.loc[primary.model.eq("PCA4 RBF")].set_index("date").loc[q4.index]
    uncoupled=primary.loc[primary.model.eq("Q4 no RZZ")].set_index("date").loc[q4.index]
    differences=(q4.hits-rbf.hits).to_numpy(dtype=int)
    slots=q4.slots.to_numpy(dtype=int)
    sensitivity={"unit":"observed date","difference_order":"coupled minus matched RBF",
                 "observed_detection_difference":int(differences.sum()),
                 "coupled_minus_uncoupled":int((q4.hits-uncoupled.hits).sum()),
                 "observed_difference_per_100_inspections":float(100*differences.sum()/slots.sum()),
                 "coupled_wins":int((differences>0).sum()),"ties":int((differences==0).sum()),
                 "coupled_losses":int((differences<0).sum()),
                 "uncertainty_note_sha256":"e5069e44ef1d23b30d416de464c3d15723358dbd8de98db87bdd5ec7ebd981a1"}
    if len(slots)>=15:
        n=len(slots)
        starts=np.random.default_rng(20261123).integers(0,n,size=(10000,(n+4)//5))
        samples=((starts[:,:,None]+np.arange(5))%n).reshape(10000,-1)[:,:n]
        den=slots[samples].sum(axis=1)
        stats=100*differences[samples].sum(axis=1)[den>0]/den[den>0]
        lo,hi=np.percentile(stats,[2.5,97.5])
        sensitivity.update(block_length=5,bootstrap_seed=20261123,replicates=10000,
                           usable_replicates=int(len(stats)),lower_percentile=float(lo),upper_percentile=float(hi))
    write_json(output/"external_primary_contrast.json",sensitivity)
    print(totals.to_string(index=False))
    print(json.dumps(sensitivity,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage",choices=["freeze","evaluate"])
    parser.add_argument("--data",type=Path, help="Local original train.csv. When omitted the pinned source is downloaded.")
    parser.add_argument("--output",type=Path,default=Path("diecasting_results"))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    raw=bytes_for_analysis(args.data)
    (freeze if args.stage=="freeze" else evaluate)(raw,args.output)


if __name__=="__main__":
    main()
