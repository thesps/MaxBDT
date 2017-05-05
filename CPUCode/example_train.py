from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import bdtToJSON as bdt2j

X, y = make_moons(noise=0.3, random_state=0)

bdt =  GradientBoostingClassifier()
X_train, X_test, y_train, y_test =\
  train_test_split(X, y, test_size=.4, random_state=42)

bdt.fit(X_train, y_train)

# write to JSON
with open('bdtjson.txt', 'w') as f:
  f.write(bdt2j.toJSON(bdt))

# write to pkl
joblib.dump(bdt, 'bdt.pkl')
