using Serialization,CategoricalArrays,DataFrames
obj = deserialize("data/object.rds")

y = obj.y
X = obj.X


