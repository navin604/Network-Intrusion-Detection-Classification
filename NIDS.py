from sklearn import metrics
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import joblib
import ipaddress
import matplotlib.pyplot as plt
from sklearn import tree
import math

# Pre-Processing
pima = pd.read_csv("data/UNSW-NB15-BALANCED-TRAIN.csv")
pima["srcip"]=[int(ipaddress.ip_address(f)) for f in pima["srcip"]] # changing ip addresses to integers
pima["dstip"]=[int(ipaddress.ip_address(f)) for f in pima["dstip"]]
pima["sport"]=[int(f) if str(f).isdigit() else -1 for f in pima["sport"]] # accounting for 0x00c from ICMP
pima["dsport"]=[int(f) if str(f).isdigit() else -1 for f in pima["dsport"]]
for feature in pima:
    pima[feature] = [f if str(f).strip() else -1 for f in pima[feature]] # filling empty spots with -1
pima["ct_ftp_cmd"]=[int(f) for f in pima["ct_ftp_cmd"]] # not sure why but ct_ftp_cmd is handled as a string?
pima["ct_flw_http_mthd"] = [f if not math.isnan(f) else -1 for f in pima["ct_flw_http_mthd"]] # empty spaces here are treated as NaN
pima["is_ftp_login"] = [f if not math.isnan(f) else -1 for f in pima["is_ftp_login"]]
pima["proto"], _ = pd.factorize(pima["proto"]) # Factorizing categorical features
pima["state"], _ = pd.factorize(pima["state"])
pima["service"], _ = pd.factorize(pima["service"])
pima["service"] = pima["service"].astype("str")
print(_)
pima["attack_cat"]= [f.strip() if type(f) == str else "None" for f in pima["attack_cat"]] # fixing empty strings in attack_cat and random spaces around categories
pima["attack_cat"]= ["Backdoors" if f == "Backdoor" else f for f in pima["attack_cat"]] # combining "Backdoor" and "Backdoors" into one
pima["attack_cat"], _ = pd.factorize(pima["attack_cat"])
pima["attack_cat"] = pima["attack_cat"].astype("str")
print(_)

# Feature selection
X = pima.drop(["attack_cat", "Label"], axis=1)
y = pima.attack_cat # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
clf = DecisionTreeClassifier(random_state=0)
rfe = RFE(clf, n_features_to_select=10)
X_train_rfe = rfe.fit_transform(X_train, y_train) # trimming features
X_val_rfe = rfe.transform(X_test)

# Training and Prediction
fit = clf.fit(X_train_rfe, y_train)
y_pred = clf.predict(X_val_rfe)
print("Num Features: ", rfe.n_features_)
print("Feature Ranking: ", rfe.ranking_)
features_selected = ""
for i in range(len(rfe.ranking_)):
    if rfe.ranking_[i] == 1:
        features_selected += pima.columns[i] + ", "
print(f"features used: {features_selected}")
print(f"ypred: {y_pred}")
print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))
print(f"macro f1 score: {metrics.f1_score(y_test, y_pred, average='macro')}")
print(f"micro f1 score: {metrics.f1_score(y_test, y_pred, average='micro')}\n")
# fig, ax = plt.subplots(figsize=(12, 12))
# plot_tree(clf, filled=True, ax=ax, feature_names=pima.columns, class_names=y_train.unique())
# plt.show()
text = tree.export_text(clf)
print(text)
joblib.dump(clf, "dt_model")


def process_args():
    print("Processing Args")


if __name__ == "__main__":
    process_args()