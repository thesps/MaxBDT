import json
import numpy as np

def toDict(tree):
  treeDict = {'features' : tree.feature.tolist(), 'thresholds' : tree.threshold.tolist(), 'values' : tree.value[:,0,0].tolist()}
  treeDict['childrenLeft'] = tree.children_left.tolist()
  treeDict['childrenRight'] = tree.children_right.tolist()
  return treeDict

def toJSON(bdt):
  ensembleDict = {'learningRate' : bdt.learning_rate, 'initPredict' : bdt.init_.predict(np.array([0]))[0][0]}
  ensembleDict['nFeatures'] = bdt.n_features
  trees = [toDict(tree[0].tree_) for tree in bdt.estimators_]
  ensembleDict['trees'] = trees
  return json.dumps(ensembleDict)

