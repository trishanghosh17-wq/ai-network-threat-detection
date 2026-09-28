"""
End-to-end training script for Network Traffic Classification (BENIGN vs DDoS).
Run: python src/train.py
"""
import os, json, joblib, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedGroupKFold, RandomizedSearchCV
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from xgboost import XGBClassifier

ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA=os.path.join(ROOT,"data","DDos.pcap_ISCX.csv")
OUT=os.path.join(ROOT,"results"); MODELS=os.path.join(ROOT,"models"); VIZ=os.path.join(ROOT,"visualizations")
os.makedirs(OUT,exist_ok=True); os.makedirs(MODELS,exist_ok=True); os.makedirs(VIZ,exist_ok=True)

df=pd.read_csv(DATA,low_memory=False)
df.columns=df.columns.str.strip()
df["Label"]=df["Label"].astype(str).str.strip()
num=df.drop(columns=["Label"]).apply(pd.to_numeric,errors="coerce")
num=num.replace([np.inf,-np.inf],np.nan)
clean=df.loc[~df.duplicated()].copy()
X=clean.drop(columns=["Label"]).apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
y=(clean["Label"].str.upper()=="DDOS").astype(int)
constant=[c for c in X.columns if X[c].nunique(dropna=False)<=1]
X=X.drop(columns=constant)

# Strict group-aware split: identical feature vectors cannot cross train/test.
groups=pd.util.hash_pandas_object(X.fillna(-999999999999),index=False).values
cv=StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=42)
tr,te=next(cv.split(X,y,groups))
Xtr,Xte=X.iloc[tr],X.iloc[te]; ytr,yte=y.iloc[tr],y.iloc[te]

# Leakage-safe feature selection on training only.
rng=np.random.RandomState(42)
idx=rng.choice(len(Xtr),min(50000,len(Xtr)),replace=False)
Xi=SimpleImputer(strategy="median").fit_transform(Xtr.iloc[idx])
mi=mutual_info_classif(Xi,ytr.iloc[idx],random_state=42)
features=pd.Series(mi,index=Xtr.columns).sort_values(ascending=False).head(30).index.tolist()

models={
"Decision Tree":DecisionTreeClassifier(random_state=42,class_weight="balanced",min_samples_leaf=2),
"Random Forest":RandomForestClassifier(n_estimators=150,random_state=42,n_jobs=-1,class_weight="balanced_subsample",max_features="sqrt"),
"Extra Trees":ExtraTreesClassifier(n_estimators=150,random_state=42,n_jobs=-1,class_weight="balanced",max_features="sqrt"),
"XGBoost":XGBClassifier(n_estimators=200,max_depth=6,learning_rate=.08,subsample=.9,colsample_bytree=.9,
 objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=-1)
}
rows=[]; fitted={}
for name,m in models.items():
    pipe=Pipeline([("imputer",SimpleImputer(strategy="median")),("model",m)])
    pipe.fit(Xtr[features],ytr)
    p=pipe.predict(Xte[features]); pr=pipe.predict_proba(Xte[features])[:,1]
    tn,fp,fn,tp=confusion_matrix(yte,p).ravel()
    rows.append({"Model":name,"Accuracy":accuracy_score(yte,p),"Precision":precision_score(yte,p),
                 "Recall":recall_score(yte,p),"F1":f1_score(yte,p),"ROC_AUC":roc_auc_score(yte,pr),
                 "TN":tn,"FP":fp,"FN":fn,"TP":tp})
    fitted[name]=pipe

vote=VotingClassifier(estimators=[("rf",fitted["Random Forest"]),("et",fitted["Extra Trees"]),("xgb",fitted["XGBoost"])],
                      voting="soft",weights=[1,1,2],n_jobs=-1)
vote.fit(Xtr[features],ytr)
p=vote.predict(Xte[features]); pr=vote.predict_proba(Xte[features])[:,1]
tn,fp,fn,tp=confusion_matrix(yte,p).ravel()
rows.append({"Model":"Soft Voting Ensemble","Accuracy":accuracy_score(yte,p),"Precision":precision_score(yte,p),
             "Recall":recall_score(yte,p),"F1":f1_score(yte,p),"ROC_AUC":roc_auc_score(yte,pr),
             "TN":tn,"FP":fp,"FN":fn,"TP":tp})

# Small randomized hyperparameter search for XGBoost using grouped CV.
dist={"model__n_estimators":[100,150,200,300],"model__max_depth":[4,6,8],
      "model__learning_rate":[.03,.05,.08,.1],"model__subsample":[.8,.9,1.0],
      "model__colsample_bytree":[.8,.9,1.0],"model__min_child_weight":[1,3,5]}
xp=Pipeline([("imputer",SimpleImputer(strategy="median")),
             ("model",XGBClassifier(objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=-1))])
search=RandomizedSearchCV(xp,dist,n_iter=6,scoring="f1",cv=cv,random_state=42,n_jobs=-1)
search.fit(Xtr[features],ytr,groups=groups[tr])
best=search.best_estimator_
p=best.predict(Xte[features]); pr=best.predict_proba(Xte[features])[:,1]
tn,fp,fn,tp=confusion_matrix(yte,p).ravel()
rows.append({"Model":"Tuned XGBoost","Accuracy":accuracy_score(yte,p),"Precision":precision_score(yte,p),
             "Recall":recall_score(yte,p),"F1":f1_score(yte,p),"ROC_AUC":roc_auc_score(yte,pr),
             "TN":tn,"FP":fp,"FN":fn,"TP":tp})

pd.DataFrame(rows).to_csv(os.path.join(OUT,"final_model_comparison_strict.csv"),index=False)
with open(os.path.join(OUT,"tuned_parameters.json"),"w") as f: json.dump(search.best_params_,f,indent=2)

# Retrain deployment model on all cleaned labeled data after evaluation.
idx=np.random.RandomState(42).choice(len(X),min(50000,len(X)),replace=False)
Xi=SimpleImputer(strategy="median").fit_transform(X.iloc[idx])
mi=mutual_info_classif(Xi,y.iloc[idx],random_state=42)
deploy_features=pd.Series(mi,index=X.columns).sort_values(ascending=False).head(30).index.tolist()
params={k.replace("model__",""):v for k,v in search.best_params_.items()}
deploy=Pipeline([("imputer",SimpleImputer(strategy="median")),
                 ("model",XGBClassifier(**params,objective="binary:logistic",eval_metric="logloss",random_state=42,n_jobs=-1))])
deploy.fit(X[deploy_features],y)
joblib.dump(deploy,os.path.join(MODELS,"ddos_xgboost_final.joblib"))
with open(os.path.join(MODELS,"features.txt"),"w") as f: f.write("\n".join(deploy_features))
print("Training complete. Final deployment model:",os.path.join(MODELS,"ddos_xgboost_final.joblib"))
