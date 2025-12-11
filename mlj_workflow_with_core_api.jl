
# 文件名：mlj_workflow_with_core_api.jl
# 策略：用MLJ管理数据，用库的核心API训练，再用MLJ评估
using Serialization,CategoricalArrays,DataFrames,Dates
obj = deserialize("data/object.rds")

y = obj.y
X = obj.X

using MLJ
ms = models(matching(X, y))

ms

using DecisionTree,MLJDecisionTreeInterface,MLJModelInterface
Tree = @load DecisionTreeClassifier pkg=DecisionTree
tree = Tree(min_samples_split=5, max_depth=4)
# a table and a vector
KNN = @load KNNClassifier pkg=NearestNeighborModels
knn = KNN()
MLJModelInterface.evaluate(knn, X, y,
         resampling=CV(nfolds=5),
         measure=[auc])