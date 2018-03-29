# pylint: disable=E1101

from sklearn.datasets import load_iris
from sklearn import tree
iris = load_iris()
test_idx = [0,50,100]

# training data
import numpy as np
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# testing data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# # Review iris properties
# print(iris.feature_names)
# print(iris.target_names)
# print(iris.data[0])
# print(iris.target[0])
# for i in range(len(iris.target)):
#     print("Example %d: label %s, features %s" % (i, iris.target_names[iris.target[i]], iris.data[i]))

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print(test_target)
print(clf.predict(test_data))

# viz code (from http://scikit-learn.org/stable/modules/tree.html#tree)
# prereq: conda install python-graphviz
import graphviz
dot_data = tree.export_graphviz(clf, 
    out_file=None,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True, rounded=True,
    impurity=False)

graph = graphviz.Source(dot_data)
graph.render("iris", cleanup=True, view=True)

# Alternative viz:
# prereq: conda install pydot
# import pydot
# (graph,) = pydot.graph_from_dot_data(dot_data.getvalue())
# prereq: cup -y Graphviz
# The following call will invoke dot.exe.
# graph.write_pdf("iris.pdf")
# Or ...
# from subprocess import call
# call(["dot", "-Tpdf", "irisTree.dot", "-o", "irisTree.pdf"])
