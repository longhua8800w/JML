
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
using LightGBM, Random, Statistics, Dates, DataFrames, Serialization
using ROCAnalysis  # 用于计算AUC

Random.seed!(42)
println("="^70)
println("🚀 LightGBM 端到端 AUC 优化工作流")
println("="^70)

# 1. 数据准备与检查
println("\n1. 📊 数据准备与检查")

# 假设 X, y 已经加载，y 是二分类向量 (0/1)
println("   原始数据检查:")
println("   - X 维度: $(size(X))")
println("   - y 长度: $(length(y))")
println("   - y 类别分布: 0=$(sum(y.==0)) ($(round(mean(y.==0)*100,digits=1))%), 1=$(sum(y.==1)) ($(round(mean(y.==1)*100,digits=1))%)")

# 转换为 MLJ 需要的格式
y_cat = coerce(y, Multiclass)  # 必须转换为 Multiclass 类型
#coerce!(X, autotype(X, :few_to_finite))

# 为 AUC 计算准备的数值标签 (1 为正类)
y_num = Vector{Float64}(y .== 1)

# 数据分割：60%训练，20%验证（调优），20%测试
train_idx, temp_idx = partition(eachindex(y_cat), 0.6, shuffle=true, rng=42)
val_idx, test_idx = partition(temp_idx, 0.5, shuffle=true, rng=42)

X_train = X[train_idx, :]; y_train = y_cat[train_idx]; y_train_num = y_num[train_idx]
X_val = X[val_idx, :];   y_val = y_cat[val_idx];   y_val_num = y_num[val_idx]
X_test = X[test_idx, :]; y_test = y_cat[test_idx]; y_test_num = y_num[test_idx]

println("\n   数据分割 (AUC优化):")
println("   - 训练集: $(length(train_idx)) 样本 (模型训练)")
println("   - 验证集: $(length(val_idx)) 样本 (参数调优)")
println("   - 测试集: $(length(test_idx)) 样本 (最终评估)")

# 2. AUC 评估函数
println("\n2. 📈 定义 AUC 评估函数")
using MLJBase
function calculate_auc(mach, X_data, y_true)
    """
    计算模型在给定数据上的 AUC
    """
    y_prob = MLJ.predict(mach, X_data)
    #  AUC
    res = MLJ.auc(y_prob, y_true)
    return res
end

# 3. 加载并配置 LightGBM 模型（AUC优化专用）
println("\n3. 🎯 配置 LightGBM (AUC优化)")

# 加载 LightGBM 分类器
LGB = @load LGBMClassifier pkg=LightGBM

# 基础模型配置 - 针对 AUC 优化
base_model = LGB(
    objective="binary",
    metric=["auc"],           # 使用 AUC 作为评估指标
    boosting="gbdt",
    verbosity=-1,           # 减少输出
    seed=42,
    
    # 处理不平衡数据的参数（如果正样本很少）
    scale_pos_weight=length(y_train_num)/(2*sum(y_train_num))
)

println("   基础模型配置:")
println("   - 目标函数: binary")
println("   - 评估指标: auc")
println("   - 提升类型: gbdt")
println("   - 随机种子: 42")
if base_model.is_unbalance
    println("   - 不平衡处理: 开启 (正样本比例=$(round(mean(y_train_num),digits=3)))")
end

# 4. 定义调优参数空间（AUC优化专用）
println("\n4. ⚙️ 定义调优参数空间")

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

# 使用随机搜索（比贝叶斯优化更稳定快速）
tuning = RandomSearch(rng=456)

println("   调优配置:")
println("   - 交叉验证: 5折")
println("   - 调优算法: 随机搜索")
println("   - 评估次数: 80次")
println("   - 优化目标: 最大化AUC")

# 6. 创建并执行调优
println("\n6. 🚀 开始 LightGBM AUC 优化调优...")
println("   开始时间: $(now())")

# 创建调优模型
tuned_model = TunedModel(
    model=base_model,
    tuning=tuning,
    resampling=cv,
    ranges=tuning_ranges,
    measure=MLJ.auc,            # 关键：优化AUC！
    n=80,                   # 评估80组参数
    acceleration=CPUThreads()  # 使用多线程加速
)

# 在验证集上执行调优（不是训练集！）
mach = machine(tuned_model, X_val, y_val)
MLJ.fit!(mach, verbosity=2)  # verbosity=2 显示进度条

println("   结束时间: $(now())")
println("   ✅ 调优完成！")

# 7. 提取和分析最优模型
println("\n7. 📊 分析调优结果")

# 获取调优报告
report_df = report(mach)
best_model = fitted_params(mach).best_model
best_auc = report_df.best_history_entry.measurement[1]

println("   🏆 最优参数组合:")
println("   - num_leaves:        $(best_model.num_leaves)")
println("   - max_depth:         $(best_model.max_depth)")
println("   - learning_rate:     $(round(best_model.learning_rate, digits=4))")
println("   - num_iterations:    $(best_model.num_iterations)")
println("   - lambda_l1:         $(round(best_model.lambda_l1, digits=4))")
println("   - lambda_l2:         $(round(best_model.lambda_l2, digits=4))")
println("   - min_data_in_leaf:  $(best_model.min_data_in_leaf)")
println("   - feature_fraction:  $(round(best_model.feature_fraction, digits=3))")
println("   - bagging_fraction:  $(round(best_model.bagging_fraction, digits=3))")
println("   - bagging_freq:      $(best_model.bagging_freq)")

println("\n   验证集性能:")
println("   - 最佳AUC: $(round(best_auc, digits=4))")

# 查看调优历史
history = report_df.history
println("   - 总评估次数: $(length(history))")

# 8. 使用最优模型在完整训练集上训练
println("\n8. 🔧 训练最终 LightGBM 模型...")

final_model = best_model
final_mach = machine(final_model, X_train, y_train)
MLJ.fit!(final_mach, verbosity=1)

println("   ✅ 最终模型训练完成")

# 9. 综合性能评估
println("\n9. 🧪 综合性能评估")

# 9.1 训练集 AUC

train_auc = calculate_auc(final_mach, X_train, y_train)
println("   训练集 AUC: $(round(train_auc, digits=4))")

# 9.2 验证集 AUC（调优时已看过，这里再确认）
val_auc = calculate_auc(final_mach, X_val, y_val)
println("   验证集 AUC: $(round(val_auc, digits=4))")

# 9.3 测试集 AUC（最重要！）
test_auc = calculate_auc(final_mach, X_test, y_test)
println("   测试集 AUC: $(round(test_auc, digits=4))")

# 9.4 准确率等其他指标（作为参考）
y_pred_test = predict_mode(final_mach, X_test)
accuracy_test = mean(y_pred_test .== y_test)
precision_test = mean(y_pred_test[y_test.=="1"] .== "1")
recall_test = mean(y_test[y_pred_test.=="1"] .== "1")

println("\n   测试集其他指标 (参考):")
println("   - 准确率:    $(round(accuracy_test*100, digits=2))%")
println("   - 精确率:    $(round(precision_test*100, digits=2))%")
println("   - 召回率:    $(round(recall_test*100, digits=2))%")

# 10. 特征重要性分析
println("\n10. 🔍 特征重要性分析")



using LightGBM  # 确保加载

# 获取 fitted_params
fp = fitted_params(final_mach)

# 从 Tuple 中提取 Estimator（LGBMClassification，即 Tuple[1]）
lgbm_estimator = fp.fitresult[1]  # Tuple 的第一个元素是 LGBMClassification (LGBMEstimator 子类型)

# 计算特征重要性（"gain" 类型；可指定迭代次数，默认所有）
importances_gain = LightGBM.gain_importance(lgbm_estimator)  # 默认所有迭代
# 或指定迭代：importances_gain = LightGBM.gain_importance(lgbm_estimator, 100)  # 基于前 100 迭代

# 获取特征名（从 X 数据中提取；假设 X 是 table）
feature_names = schema(X).names  # e.g., [:feat1, :feat2, ...]

# 输出排序后的重要性
sorted_indices = sortperm(importances_gain, rev=true)  # 降序
println("特征重要性 (基于 Gain):")
for i in sorted_indices
    println("  特征 $(feature_names[i]): $(importances_gain[i])")
end

# 如果想用 "split" 类型（基于分裂次数）：
importances_split = LightGBM.split_importance(lgbm_estimator)
# 然后同样排序输出

# 可视化（可选，用 Plots.jl）
using Plots

feature_labels = String.(collect(feature_names[sorted_indices])) 
bar(feature_labels, importances_gain[sorted_indices], 
    title="LightGBM Feature Importances (Gain)", 
    xlabel="Features", ylabel="Importance", orientation=:h)

# 11. 模型保存
println("\n11. 💾 保存模型与结果")

# 保存最终模型
MLJ.save("mdls/lightgbm_auc_optimized_final.jlso", final_mach)

# 保存调优历史（包含所有尝试的参数组合）
tuning_history = Dict(
    :best_model => best_model,
    :best_auc => best_auc,
    :test_auc => test_auc,
    :feature_importance => fi,
    :all_history => [(h.model, h.measurement[1]) for h in history]
)


# =====================================================
# 关键指标总结
# =====================================================
println("\n📊 关键指标总结:")
println("   • 验证集最佳AUC: $(round(best_auc, digits=4))")
println("   • 测试集AUC:     $(round(test_auc, digits=4))")
println("   • 测试集准确率:  $(round(accuracy_test*100, digits=2))%")
println("   • 特征数量:      $(ncol(X))")
println("   • 最优迭代次数:  $(best_model.num_iterations)")
println("   • 学习率:        $(round(best_model.learning_rate, digits=4))")