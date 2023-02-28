import sys
import pickle
from sklearn import metrics
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import joblib
import ipaddress
import matplotlib.pyplot as plt
from sklearn import tree
import math

tasks = ["Label", "attack_cat"]
classifier = ["decision_tree", "nav_classifier", "sgd"]


def main():
    file, classification_method, task, model = process_args(sys.argv[1:])
    validate_args(classification_method, task)
    pima = pre_processing(file)
    if classification_method == "decision_tree":
        decision_tree(pima)
    elif classification_method == "nav_classifier":
        print("You can become a Data Scientist")
    elif classification_method == "sgd":
        stochastic_gradient_descent(pima, task, model)
    else:
        print("No classifier specified")


def decision_tree(pima):
    print("DECISION")
    # Feature selection
    X = pima.drop(["attack_cat", "Label"], axis=1)
    y = pima.attack_cat  # Target variable
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
    clf = DecisionTreeClassifier(random_state=0)
    rfe = RFE(clf, n_features_to_select=10)
    X_train_rfe = rfe.fit_transform(X_train, y_train)  # trimming features
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
    print(metrics.classification_report(y_test, y_pred, digits=6))
    # fig, ax = plt.subplots(figsize=(12, 12))
    # plot_tree(clf, filled=True, ax=ax, feature_names=pima.columns, class_names=y_train.unique())
    # plt.show()
    text = tree.export_text(clf)
    print(text)
    joblib.dump(clf, "dt_model")


def stochastic_gradient_descent(pima, task, model):
    X = pima.drop(["attack_cat", "Label"], axis=1)
    y = pima[task]

    if not model:
        # Split data into test and train
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

        # Create Classifier and implement scaling
        clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=10000, loss="modified_huber"))

        # Using Factor Analysis analysis technique
        fa = FactorAnalysis()
        X_train_cca = fa.fit_transform(X_train, y_train)

        # Fit model
        clf.fit(X_train_cca, y_train)

        # Save model
        with open(f"sgd_{task}.sav", 'wb') as model_file:
            pickle.dump([fa, clf], model_file)

    else:
        try:
            X_test = X
            y_test = y

            # Load model
            with open(model, 'rb') as model_file:
                fa, clf = pickle.load(model_file)
        except FileNotFoundError:
            print("Model not found. Exiting...")
            return

    # Transform test data using Factor Analysis
    X_test_cca = fa.transform(X_test)
    # Predict using model
    y_pred = clf.predict(X_test_cca)

    print(metrics.classification_report(y_test, y_pred, digits=6))
    print("Accuracy: {:.2f}%".format(metrics.accuracy_score(y_test, y_pred) * 100))
    print(f"macro f1 score: {metrics.f1_score(y_test, y_pred, average='macro')}")
    print(f"micro f1 score: {metrics.f1_score(y_test, y_pred, average='micro')}\n")


def pre_processing(csv):
    pima = pd.read_csv(csv)
    pima["srcip"] = [int(ipaddress.ip_address(f)) for f in pima["srcip"]]  # changing ip addresses to integers
    pima["dstip"] = [int(ipaddress.ip_address(f)) for f in pima["dstip"]]
    pima["sport"] = [int(f) if str(f).isdigit() else -1 for f in pima["sport"]]  # accounting for 0x00c from ICMP
    pima["dsport"] = [int(f) if str(f).isdigit() else -1 for f in pima["dsport"]]
    for feature in pima:
        pima[feature] = [f if str(f).strip() else -1 for f in pima[feature]]  # filling empty spots with -1
    pima["ct_ftp_cmd"] = [int(f) for f in pima["ct_ftp_cmd"]]  # not sure why but ct_ftp_cmd is handled as a string?
    pima["ct_flw_http_mthd"] = [f if not math.isnan(f) else -1 for f in
                                pima["ct_flw_http_mthd"]]  # empty spaces here are treated as NaN
    pima["is_ftp_login"] = [f if not math.isnan(f) else -1 for f in pima["is_ftp_login"]]
    pima["proto"], _ = pd.factorize(pima["proto"])  # Factorizing categorical features
    pima["state"], _ = pd.factorize(pima["state"])
    pima["service"], _ = pd.factorize(pima["service"])
    pima["service"] = pima["service"].astype("str")
    print(_)
    pima["attack_cat"] = [f.strip() if type(f) == str else "None" for f in
                          pima["attack_cat"]]  # fixing empty strings in attack_cat and random spaces around categories
    pima["attack_cat"] = ["Backdoors" if f == "Backdoor" else f for f in
                          pima["attack_cat"]]  # combining "Backdoor" and "Backdoors" into one
    pima["attack_cat"], _ = pd.factorize(pima["attack_cat"])
    pima["attack_cat"] = pima["attack_cat"].astype("str")
    print(_)
    return pima


def validate_args(classifier, task):
    if classifier not in classifier:
        sys.exit("Invalid classification method")
    if task not in tasks:
        sys.exit("Invalid task")


def process_args(args):
    if len(args) == 3:
        args.append(None)
    return args[0], args[1], args[2], args[3]


if __name__ == "__main__":
    main()

