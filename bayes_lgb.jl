
# 文件名：mlj_workflow_with_core_api.jl
# 策略：用MLJ管理数据，用库的核心API训练，再用MLJ评估
using Serialization,CategoricalArrays,DataFrames,Dates
obj = deserialize("data/object.rds")

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

# 数据分割：60%训练，20%验证（调优），20%测试
train_idx, temp_idx = partition(eachindex(y_cat), 0.6, shuffle=true, rng=42)
val_idx, test_idx = partition(temp_idx, 0.5, shuffle=true, rng=42)

X_train = X[train_idx, :]; y_train = y_cat[train_idx]
X_val = X[val_idx, :];   y_val = y_cat[val_idx]
X_test = X[test_idx, :]; y_test = y_cat[test_idx]


# 2. AUC 评估函数

using MLJBase
function calculate_auc(mach, X_data, y_true)
    y_prob = MLJ.predict(mach, X_data)
    res = MLJ.auc(y_prob, y_true)
    return res
end


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



# 4. 定义调优参数空间（AUC优化专用）

tuning_ranges = [
    # 核心复杂度参数
    range(base_model, :num_leaves, lower=20, upper=150, scale=:log),  # 叶子数量
    range(base_model, :max_depth, lower=3, upper=12),                 # 树的最大深度
    
    # 学习过程参数
    range(base_model, :learning_rate, lower=0.01, upper=0.3, scale=:log),
    range(base_model, :num_iterations, lower=50, upper=500, scale=:log),
    
    # 正则化参数（防止过拟合，提升AUC）
    range(base_model, :lambda_l1, lower=0.0, upper=10.0, scale=:log),  # L1正则化
    range(base_model, :lambda_l2, lower=0.0, upper=10.0, scale=:log),  # L2正则化
    range(base_model, :min_data_in_leaf, lower=10, upper=100, scale=:log),
    
    # 随机化参数（提升模型鲁棒性）
    range(base_model, :feature_fraction, lower=0.6, upper=1.0),  # 特征采样比例
    range(base_model, :bagging_fraction, lower=0.6, upper=1.0),  # 数据采样比例
    range(base_model, :bagging_freq, lower=1, upper=10)          # bagging频率
]

println("   调优参数 (9个关键参数):")
for (i, r) in enumerate(tuning_ranges)
    scale_info = r.scale == :log ? "[对数尺度]" : ""
    println("   $(lpad(i,2)). $(rpad(string(r.field), 20)): $(r.lower) → $(r.upper) $scale_info")
end






# 5. 配置 TreeParzen 贝叶斯优化
println("\n5. 🎯 配置 TreeParzen 贝叶斯优化")

using TreeParzen

# 将 MLJ ranges 转换为 TreeParzen 的先验分布
priors = Dict{Symbol, Any}(
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

println("已创建 $(length(priors)) 个参数的先验分布")

# 查看创建的先验
println("\n先验分布配置:")
for (key, prior) in priors
    println("  $key: $prior")
end


# 创建 TreeParzen 调优器
# 创建 TreeParzen 调优器
# 5. 配置 TreeParzen 贝叶斯优化
println("\n5. 🎯 配置 TreeParzen 贝叶斯优化")

using TreeParzen

NUM_CV_FOLDS = 4
PCT_TRAIN_DATA = 0.75
NUM_TP_ITER_SMALL = 25
NUM_TP_ITER_LARGE = 250

tuning = MLJTuning.TunedModel(
    model=base_model,
    range=priors,
    tuning=MLJTreeParzenTuning(),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measure=MLJ.auc,
)



mach = MLJ.machine(tuning, X_train, y_train)

println("开始时间: $(now())")
MLJ.fit!(mach, verbosity=2)
println("结束时间: $(now())")



best_model = MLJ.fitted_params(mach).best_model

suggestion = Dict(key => getproperty(best_model, key) for key in keys(priors))

search = MLJTreeParzenSpace(priors, suggestion)

tuning2 = MLJTuning.TunedModel(
    model=base_model,
    range=search,
    tuning=MLJTreeParzenTuning(;random_trials=3),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measure=MLJ.auc,
)


mach2 = MLJ.machine(tuning2, X_train, y_train)

println("开始时间: $(now())")
MLJ.fit!(mach2, verbosity=2)
println("结束时间: $(now())")





tuning21 = MLJTuning.TunedModel(
    model=base_model,
    range=search,
    tuning=MLJTreeParzenTuning(;random_trials=3, max_simultaneous_draws=2, linear_forgetting=50),
    n=NUM_TP_ITER_SMALL,
    resampling=MLJ.CV(nfolds=NUM_CV_FOLDS),
    measure=MLJ.auc,
)




mach21 = MLJ.machine(tuning21, X_train, y_train)

println("开始时间: $(now())")
MLJ.fit!(mach, verbosity=2)
println("结束时间: $(now())")
