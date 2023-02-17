from sklearn import metrics
from sklearn.feature_selection import RFE
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split


# Pre-Processing
pima = pd.read_csv("data/UNSW-NB15-BALANCED-TRAIN.csv")
for f in pima.columns:
    pima[f], _ = pd.factorize(pima[f]) # Factorizing columns
X = pima.drop(["attack_cat", "Label"], axis=1) # Features
y = pima.attack_cat # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)


clf = DecisionTreeClassifier(random_state=0)
rfe = RFE(clf, n_features_to_select=10) # Feature Selection

X_train_rfe = rfe.fit_transform(X_train, y_train) # trimming features
X_val_rfe = rfe.transform(X_test)

fit = clf.fit(X_train_rfe, y_train) # training

y_pred = clf.predict(X_val_rfe)
print("Num Features: ", rfe.n_features_)
print("Feature Ranking: ", rfe.ranking_)
print("Accuracy: {:.2f}%\n".format(metrics.accuracy_score(y_test, y_pred) * 100))
