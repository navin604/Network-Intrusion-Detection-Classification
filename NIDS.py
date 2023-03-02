import pickle
import sys
import time
from sklearn import metrics
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import ipaddress
import math
import warnings
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

tasks = ["Label", "attack_cat"]
classifiers = ["decision_tree", "svm", "sgd"]


def main():
    file, classification_method, task, model = process_args(sys.argv[1:])
    validate_args(classification_method, task)
    pima = pre_processing(file)

    if classification_method == "decision_tree":
        decision_tree(pima, task, model)
    elif classification_method == "svm":
        svm(pima, task, model)
    elif classification_method == "sgd":
        stochastic_gradient_descent(pima, task, model)
    else:
        print("No classifier specified")


def svm(pima, task, model):
    # Loading variables into Pandas dataframe without columns
    X = pima.drop(["attack_cat", "Label"], axis=1)
    # Target data
    y = pima[task]

    if model:
        X_test = X
        y_test = y
        file = open(model, 'rb')
        pca, clf = pickle.load(file)

    else:
        # Divide data into training and testing
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        pca = PCA()
        X_train = pca.fit_transform(X_train)
        # Instantiate SVC model
        clf = make_pipeline(StandardScaler(), LinearSVC(multi_class="ovr", dual=False))
        clf.fit(X_train, y_train)
        filename = "svm_" + task + '.sav'
        file = open(filename, 'wb')
        pickle.dump([pca, clf], file)

    file.close()
    X_test = pca.transform(X_test)
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))
    print(f"micro f1 score: {metrics.f1_score(y_test, y_pred, average='micro')}\n")


def decision_tree(pima, task, model):
    X = pima.drop(["attack_cat", "Label"], axis=1)
    y = pima[task]

    if model:
        X_test = X
        y_test = y
        fo = open(model, 'rb')
        rfe, clf = pickle.load(fo)

    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
        rfe = RFE(DecisionTreeClassifier(), n_features_to_select=10)
        X_train = rfe.fit_transform(X_train, y_train)
        clf = DecisionTreeClassifier(random_state=0)
        clf.fit(X_train, y_train)
        filename = "decision_tree_" + task + '.sav'
        fo = open(filename, 'wb')
        pickle.dump([rfe, clf], fo)

    fo.close()
    X_test = rfe.transform(X_test)
    y_pred = clf.predict(X_test)

    print(f"Number of Features: {rfe.n_features_}")
    print(f"Features used: {rfe.get_feature_names_out()}")
    print("Classifier: DecisionTree")
    if task == "attack_cat":
        print(classification_report(y_test, y_pred, target_names=attack_labels))
    else:
        print(classification_report(y_test, y_pred))


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

        # Save model by writing bytes
        with open(f"sgd_{task}.sav", 'wb') as model_file:
            pickle.dump([fa, clf], model_file)

    else:
        try:
            X_test = X
            y_test = y

            # Load model by reading bytes
            with open(model, 'rb') as model_file:
                fa, clf = pickle.load(model_file)
        except FileNotFoundError:
            print("Model not found. Exiting...")
            return
        except EOFError:
            print("Model is incorrect or is corrupt. Exiting...")
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
    pima["proto"], _ = pd.factorize(pima["proto"], sort=True)  # Factorizing categorical features
    pima["state"], _ = pd.factorize(pima["state"], sort=True)
    pima["service"], _ = pd.factorize(pima["service"], sort=True)
    pima["service"] = pima["service"].astype("str")
    pima["attack_cat"] = [f.strip() if type(f) == str else "None" for f in
                          pima["attack_cat"]]  # fixing empty strings in attack_cat and random spaces around categories
    pima["attack_cat"] = ["Backdoors" if f == "Backdoor" else f for f in
                          pima["attack_cat"]]  # combining "Backdoor" and "Backdoors" into one
    global attack_labels
    pima["attack_cat"], attack_labels = pd.factorize(pima["attack_cat"], sort=True)
    pima["attack_cat"] = pima["attack_cat"].astype("str")
    return pima


def validate_args(classifier, task):
    if classifier not in classifiers:
        sys.exit("Invalid classification method")
    if task not in tasks:
        sys.exit("Invalid task")


def process_args(args):
    if len(args) == 3:
        args.append(None)
    return args[0], args[1], args[2], args[3]


if __name__ == "__main__":
    main()
