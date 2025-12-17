
# 文件名：mlj_workflow_with_core_api.jl
# 策略：用MLJ管理数据，用库的核心API训练，再用MLJ评估
using Serialization,CategoricalArrays,DataFrames,Dates



obj = deserialize("data/xy")

y = obj.y
X = obj.X


# =====================================================
# LightGBM 端到端 AUC 优化工作流
# =====================================================
using MLJ, MLJTuning
using LightGBM, Random, Statistics
using ROCAnalysis  # 用于计算AUC

Random.seed!(42)

# 转换为 MLJ 需要的格式
y_cat = coerce(y, Multiclass)  # 必须转换为 Multiclass 类型
#coerce!(X, autotype(X, :few_to_finite))

PCT_TRAIN_DATA = 0.75
# 数据分割：训练，测试 tscv 不能洗牌 需要保持顺序
train_idx, test_idx = partition(eachindex(y_cat), PCT_TRAIN_DATA)

X_train = X[train_idx, :]; y_train = y_cat[train_idx]
X_test = X[test_idx, :]; y_test = y_cat[test_idx]


# 加载 LightGBM 分类器
LGB = @load LGBMClassifier pkg=LightGBM

# 基础模型配置 - 针对 AUC 优化
base_model = LGB(
    objective="binary",
    metric=["auc"],           # 使用 AUC 作为评估指标
    boosting="gbdt",
    verbosity=-1,           # 减少输出
    seed=42,
    is_unbalance=true       # 处理类别不平衡
)



# 5. 配置 TreeParzen 贝叶斯优化
println("\n5. 🎯 配置 TreeParzen 贝叶斯优化")

using TreeParzen

# 将 MLJ ranges 转换为 TreeParzen 的先验分布
space = Dict{Symbol, Any}(
    # 迭代次数（对应num_round）
    :num_iterations => TreeParzen.HP.QuantUniform(:num_iterations, 50.0, 500.0, 1.0),
    
    # 学习率（对应eta）
    :learning_rate => TreeParzen.HP.LogUniform(:learning_rate, log(0.01), log(0.3)),
    
    # 叶子数量（LightGBM特有，对应max_depth但不同）
    :num_leaves => TreeParzen.HP.QuantUniform(:num_leaves, 20.0, 150.0, 1.0),
    
    # 最大深度
    :max_depth => TreeParzen.HP.QuantUniform(:max_depth, 3.0, 12.0, 1.0),
    
    # L1正则化（对应alpha）
    :lambda_l1 => TreeParzen.HP.LogUniform(:lambda_l1, log(0.001), log(10.0)),
    
    # L2正则化（对应lambda）
    :lambda_l2 => TreeParzen.HP.LogUniform(:lambda_l2, log(0.001), log(10.0)),
    
    # 最小叶子样本数（对应min_child_weight但不同）
    :min_data_in_leaf => TreeParzen.HP.QuantUniform(:min_data_in_leaf, 10.0, 100.0, 1.0),
    
    # 特征采样比例
    :feature_fraction => TreeParzen.HP.Uniform(:feature_fraction, 0.6, 1.0),
    
    # 数据采样比例
    :bagging_fraction => TreeParzen.HP.Uniform(:bagging_fraction, 0.6, 1.0),
    
    # bagging频率
    :bagging_freq => TreeParzen.HP.QuantUniform(:bagging_freq, 1.0, 10.0, 1.0)
)

println("已创建 $(length(space)) 个参数的先验分布")

# 查看创建的先验
println("\n先验分布配置:")
for (key, prior) in space
    println("  $key: $prior")
end


# 创建 TreeParzen 调优器
# 创建 TreeParzen 调优器
# 5. 配置 TreeParzen 贝叶斯优化
println("\n5. 🎯 配置 TreeParzen 贝叶斯优化")


NUM_CV_FOLDS = 5
NUM_CV_REPEATS = 6
NUM_TP_ITER_SMALL = 30
NUM_TP_ITER_LARGE = length(space)*50


using ComputationalResources
tuning_tscv = MLJTuning.TunedModel(
    model=base_model,
    range=space,
    tuning=MLJTreeParzenTuning(max_simultaneous_draws=4),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.TimeSeriesCV(nfolds=5),
    repeats=NUM_CV_REPEATS,
    measure=MLJ.auc,
    acceleration=ComputationalResources.CPUProcesses(),
)

mach_tscv = MLJ.machine(tuning_tscv, X_train, y_train)

println("开始时间: $(now())")
MLJ.fit!(mach_tscv, verbosity=2)
println("结束时间: $(now())")