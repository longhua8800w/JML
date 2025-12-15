
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

# 5. 配置重复CV调优策略
println("\n5. 🔄 配置重复CV调优策略")

# 使用5折交叉验证，配合随机搜索
cv = CV(nfolds=5, shuffle=true, rng=123)

# 查看 MLJ 官方支持的优化策略
using MLJTuning, MLJBalancing

println("MLJ 官方支持的调优策略:")
strategies = MLJTuning.TuningStrategy()
for s in strategies
    println("  - $s")
end