"""Verify the historical fit, freeze future scores, then open outcomes in a separate stage."""
from pathlib import Path
from urllib.parse import quote
from external_diecasting import *

FOLLOW_COMMIT="ad2b1147ea73fcdd54fed0507ed2f73da3c652b9"
FOLLOW_BLOB="fa4909461f32ee163686b04ac51e7c9f95f6b753"
FOLLOW_SHA="1894052d1091642a110cd76710e9d8bd995d4a30188a75d6c4fe184bf82dcf4f"
FOLLOW_PROTOCOL="2276c711d441a4c2986f9604f94a682f6070d874d8180a530eef961a571389f2"
FULL_URL=f"https://raw.githubusercontent.com/greatsangho/2024-Kamp-Competition/{FOLLOW_COMMIT}/"+quote("data/경진대회용 주조 공정최적화 데이터셋.csv",safe="/")


def full_source(file):
    if file:
        data=Path(file).read_bytes()
    else:
        with urlopen(Request(FULL_URL,headers={"User-Agent":"QML-inspection-reproduction"}),timeout=120) as response:
            data=response.read()
    git_sha=hashlib.sha1(f"blob {len(data)}\0".encode("ascii")+data).hexdigest()
    if len(data)!=21479807 or digest(data)!=FOLLOW_SHA or git_sha!=FOLLOW_BLOB:
        raise ValueError("Full-source file identity mismatch.")
    return data


def rebuild_predict(train_x,y):
    process=make_pipeline(SimpleImputer(strategy="median"),VarianceThreshold(),StandardScaler())
    tx=process.fit_transform(train_x)
    pca=PCA(n_components=4,svd_solver="full")
    pc=pca.fit_transform(tx)
    scale=0.5/np.sqrt(np.mean(pc**2))
    classical=SVC(C=1,class_weight="balanced",kernel="rbf",gamma=1/(pc.shape[1]*pc.var())).fit(pc,y)
    linear=LogisticRegression(C=1,class_weight="balanced",solver="lbfgs",max_iter=2000,tol=1e-8).fit(pc,y)
    kernels={}
    for name,coupled in (("Q4 + RZZ",True),("Q4 no RZZ",False)):
        train_states=four_qubit_states(pc*scale,coupled)
        model=SVC(C=1,kernel="precomputed",class_weight="balanced").fit(gram(train_states,train_states),y)
        kernels[name]=(coupled,train_states,model)
    def predict(x):
        pv=pca.transform(process.transform(x))
        scores={"PCA4 RBF":classical.decision_function(pv),"PCA4 logistic":linear.decision_function(pv),
                "Chronology only":-np.arange(len(pv),dtype=float)}
        for name,(coupled,train_states,model) in kernels.items():
            sv=four_qubit_states(pv*scale,coupled)
            scores[name]=np.concatenate([model.decision_function(gram(sv[i:i+192],train_states)) for i in range(0,len(sv),192)])
        return scores
    return predict


def freeze_follow_on(original,full,first_output,out):
    first=json.loads((first_output/"external_freeze.json").read_text(encoding="utf-8"))
    enc,sensors,timecol,label,mold=schema(original)
    ec,s2,t2,l2,m2=schema(full)
    assert (sensors,timecol,label,mold)==(s2,t2,l2,m2) and sensors==first["input_sensors"]
    historic_y=targets(original,enc,label)
    # Read only previously evaluated label values from the longer source.
    old_from_full=pd.to_numeric(pd.read_csv(BytesIO(full),encoding=ec,usecols=[label],nrows=len(historic_y),dtype=str)[label],errors="coerce")
    joint=historic_y.isin([0,1])&old_from_full.isin([0,1])
    np.testing.assert_array_equal(historic_y[joint].to_numpy(),old_from_full[joint].to_numpy())
    unpaired=historic_y.index[~joint].tolist()
    assert unpaired==[19327] and historic_y.loc[19327]==0 and pd.isna(old_from_full.loc[19327])
    a=pd.read_csv(BytesIO(original),encoding=enc,usecols=[timecol]+sensors,dtype=str)
    b=pd.read_csv(BytesIO(full),encoding=ec,usecols=[timecol]+sensors,dtype=str)
    pd.testing.assert_frame_equal(a,b.iloc[:len(a)].copy())
    times=pd.to_datetime(a[timecol],format="mixed",errors="coerce")
    eligible=times.notna()&historic_y.isin([0,1])
    order=times[eligible].sort_values(kind="stable").index
    earlier=order[times[order].dt.normalize()<=pd.Timestamp(first["train_end"])]
    ids=set(sorted(earlier,key=lambda x:hashlib.sha256(str(x).encode("ascii")).digest())[:2048])
    chosen=pd.Index([idx for idx in earlier if idx in ids])
    assert digest(np.asarray(chosen,dtype="<i8").tobytes())==first["sampled_train_ids_sha256"]
    historical=pd.read_csv(first_output/"frozen_scores.csv",float_precision="round_trip")
    assert not set(unpaired).intersection(chosen) and not set(unpaired).intersection(historical.source_row)
    assert digest((first_output/"frozen_scores.csv").read_bytes())==first["scores_sha256"]
    full_times=pd.to_datetime(b[timecol],format="mixed",errors="coerce")
    later=full_times[full_times.dt.normalize()>pd.Timestamp(first["test_end"])].sort_values(kind="stable").index
    x=a[sensors].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    new_x=b.loc[later,sensors].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
    with threadpool_limits(limits=1):
        model=rebuild_predict(x.loc[chosen],historic_y.loc[chosen].to_numpy(dtype=np.int8))
        reproduced=model(x.loc[historical.source_row])
        for name in MODELS:
            np.testing.assert_array_equal(reproduced[name],historical[name].to_numpy())
        later_scores=model(new_x)
    scored=pd.DataFrame({"source_row":later.to_numpy(),"date":full_times[later].dt.strftime("%Y-%m-%d").to_numpy()})
    for name in MODELS:
        scored[name]=later_scores[name]
        assert np.isfinite(scored[name]).all()
    scored.to_csv(out/"frozen_candidate_scores.csv",index=False,float_format="%.17g",lineterminator="\n")
    info={"protocol_sha256":FOLLOW_PROTOCOL,"source_sha256":digest(full),"git_blob":FOLLOW_BLOB,
          "source_rows":len(b),"old_source_rows":len(a),"raw_candidate_rows":len(scored),"candidate_dates":scored.date.nunique(),
          "start":scored.date.min(),"end":scored.date.max(),"invalid_timestamps_full_file":int(full_times.isna().sum()),
          "last_date_suffix_rows_excluded":int(sum(full_times.iloc[len(a):].dt.normalize()==pd.Timestamp(first["test_end"]))),
          "prefix_labels_checked":len(historic_y),"prefix_binary_pairs_matching":int(joint.sum()),
          "prefix_binary_pair_conflicts":0,"prefix_full_file_missing_target_rows":unpaired,
          "unpaired_target":{"source_row":19327,"registration_time":times.loc[19327].isoformat(),
                             "original_mirror_label":0,"full_source_label":None,
                             "used_in_fit":False,"used_in_first_test":False},
          "first_test_score_values_verified":len(historical)*len(MODELS),
          "sampled_train_ids_sha256":first["sampled_train_ids_sha256"],"previous_score_digest":first["scores_sha256"],
          "frozen_candidate_score_sha256":digest((out/"frozen_candidate_scores.csv").read_bytes()),
          "retained_sensors":first["retained_sensors"],"encoding":ec,"label_field":l2,"matched_reference_scores":"bit-for-bit"}
    write_json(out/"followon_candidate_freeze.json",info)
    print("New period candidate scores frozen:",info["raw_candidate_rows"],"rows,",info["candidate_dates"],"dates,",info["frozen_candidate_score_sha256"])


def evaluate_follow_on(full,out):
    info=json.loads((out/"followon_candidate_freeze.json").read_text(encoding="utf-8"))
    assert info["source_sha256"]==digest(full) and info["frozen_candidate_score_sha256"]==digest((out/"frozen_candidate_scores.csv").read_bytes())
    scored=pd.read_csv(out/"frozen_candidate_scores.csv",float_precision="round_trip")
    # Outcome validity determines eligible capacity only after all candidate scores are fixed.
    valid=targets(full,info["encoding"],info["label_field"]).loc[scored.source_row].isin([0,1]).to_numpy()
    eligible=scored.loc[valid].reset_index(drop=True)
    if not len(eligible):
        write_json(out/"followon_no_eligible_outcomes.json",{"eligible_rows":0,"raw_candidate_rows":len(scored),"test_dates_with_scores":scored.date.nunique(),"recall":None})
        print("None of the candidate records has a released binary target; recall is undefined.")
        return
    eligible.to_csv(out/"frozen_scores.csv",index=False,float_format="%.17g",lineterminator="\n")
    groups=[r.index.to_numpy() for _,r in eligible.groupby("date",sort=True)]
    sizes=np.array([len(r) for r in groups],dtype=int)
    picks=[]
    for policy in POLICIES:
        for rows,q in zip(groups,quotas(sizes,policy)):
            for name in MODELS:
                order=np.argsort(-eligible[name].to_numpy()[rows],kind="stable")
                picks.extend((policy,name,eligible.date.iloc[r],int(eligible.source_row.iloc[r])) for r in rows[order[:q]])
    pd.DataFrame(picks,columns=["policy","model","date","source_row"]).to_csv(out/"frozen_inspections.csv",index=False,lineterminator="\n")
    info.update(eligible_rows=len(eligible),eligible_dates=eligible.date.nunique(),removed_invalid_label=int((~valid).sum()),
                scores_sha256=digest((out/"frozen_scores.csv").read_bytes()),selections_sha256=digest((out/"frozen_inspections.csv").read_bytes()))
    write_json(out/"external_freeze.json",info)
    evaluate(full,out,protocol_sha=FOLLOW_PROTOCOL)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage",choices=("freeze","evaluate"))
    p.add_argument("--data",type=Path)
    p.add_argument("--full-data",type=Path)
    p.add_argument("--original-output",type=Path,default=Path("diecasting_results"))
    p.add_argument("--output",type=Path,default=Path("diecasting_followon_results"))
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    full=full_source(args.full_data)
    if args.stage=="freeze":
        freeze_follow_on(bytes_for_analysis(args.data),full,args.original_output,args.output)
    else:
        evaluate_follow_on(full,args.output)
