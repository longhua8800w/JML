using DataFrames, CSV, Statistics


# 加载主表（假设文件在当前目录）
train = CSV.read("raw_data/application_train.csv", DataFrame)
test = CSV.read("raw_data/application_test.csv", DataFrame)

# 加载辅助表（类似方式加载其他表）
bureau = CSV.read("raw_data/bureau.csv", DataFrame)
bureau_balance = CSV.read("raw_data/bureau_balance.csv", DataFrame)
prev_app = CSV.read("raw_data/previous_application.csv", DataFrame)
pos_cash = CSV.read("raw_data/POS_CASH_balance.csv", DataFrame)
installments = CSV.read("raw_data/installments_payments.csv", DataFrame)
credit_card = CSV.read("raw_data/credit_card_balance.csv", DataFrame)


#= application_train/test (SK_ID_CURR)
├── bureau (SK_ID_CURR → 多条SK_ID_BUREAU)
│   └── bureau_balance (SK_ID_BUREAU → 多条每月记录)
└── previous_application (SK_ID_CURR → 多条SK_ID_PREV)
    ├── POS_CASH_balance (SK_ID_PREV → 多条每月记录)
    ├── installments_payments (SK_ID_PREV → 多条付款记录)
    └── credit_card_balance (SK_ID_PREV → 多条每月记录)
 =#



using Pipe  
# 聚合bureau_balance到SK_ID_BUREAU

bureau_balance_agg = @pipe bureau_balance |>
    groupby(_, :SK_ID_BUREAU) |>
    combine(_,
        :MONTHS_BALANCE => mean => :BB_MONTHS_BALANCE_MEAN,
        :MONTHS_BALANCE => maximum => :BB_MONTHS_BALANCE_MAX,
        :STATUS => (x -> count(==(x), "C")) => :BB_STATUS_C_COUNT
    )

# 合并到bureau
bureau = @pipe bureau |>
leftjoin(_, bureau_balance_agg, on=:SK_ID_BUREAU)


# 聚合bureau到SK_ID_CURR
bureau_agg =@pipe bureau |>
 groupby(_, :SK_ID_CURR) |>
 combine(_,
    :CREDIT_ACTIVE => (x -> count(==(x, "Active"))) => :BUREAU_ACTIVE_COUNT,
    :AMT_CREDIT_SUM => mean => :BUREAU_CREDIT_SUM_MEAN,
    :AMT_CREDIT_SUM_DEBT => sum => :BUREAU_DEBT_SUM,
    # 添加时间序列特征，如最近信用天数：:DAYS_CREDIT => maximum => :BUREAU_DAYS_CREDIT_MAX
    # 包括从bureau_balance继承的特征
)

# 现在bureau_agg的粒度是SK_ID_CURR


# 先聚合子表到SK_ID_PREV（例如POS_CASH_balance）
pos_cash_agg = @pipe pos_cash |>
groupby(_, :SK_ID_PREV) |>  
combine(_,
    :MONTHS_BALANCE => mean => :POS_MONTHS_BALANCE_MEAN,
    :SK_DPD => sum => :POS_DPD_SUM  # 逾期天数总和
)

# 合并到prev_app
prev_app =  @pipe prev_app|>
leftjoin(_, pos_cash_agg, on=:SK_ID_PREV)

# 类似处理installments_payments和credit_card_balance
# ... (重复类似代码，聚合到SK_ID_PREV)

# 然后聚合prev_app到SK_ID_CURR
prev_app_agg = @pipe prev_app |>
groupby(_, :SK_ID_CURR) |>
combine(_,
    :AMT_APPLICATION => mean => :PREV_AMT_APP_MEAN,
    :NAME_CONTRACT_STATUS => (x -> count(==(x, "Approved"))) => :PREV_APPROVED_COUNT,
    # 包括子表特征，如:POS_DPD_SUM => mean => :PREV_POS_DPD_MEAN
)


# 合并到train（左连接，确保所有SK_ID_CURR保留）
wide_train = @pipe train |>
leftjoin(_, bureau_agg, on=:SK_ID_CURR) |>  
leftjoin(_, prev_app_agg, on=:SK_ID_CURR) |>
DataFrames.transform(_, names(_) .=> ByRow(x -> ismissing(x) ? 0 : x) .=> names(_))
# 处理缺失值（示例：填充为0）


# 对test做相同操作（无TARGET）
wide_test = @pipe test |>
leftjoin(_, bureau_agg, on=:SK_ID_CURR) |>  
leftjoin(_, prev_app_agg, on=:SK_ID_CURR) |>
DataFrames.transform(_, names(_) .=> ByRow(x -> ismissing(x) ? 0 : x) .=> names(_))
# 处理缺失值（示例：填充为0）



using  ScientificTypes, MLJ, CategoricalArrays

# 1. 先把 TARGET 正确处理成二分类
coerce!(wide_train, :TARGET => OrderedFactor{2})

# 2. 重新获取 schema（防止之前没 coerce 完）
sch = schema(wide_train)

# 3. 分类所有列
low_card_textual = String[]   # 类别数 <=10 的纯 Textual 列（要 onehot）
high_card_textual = String[]  # 类别数 >10 的纯 Textual 列（要剥离）
mixed_textual = String[]      # Union{Count,Textual} 列（也剥离）
numeric_cols = String[]       # 纯数值列（后面强制 Continuous）

for (col, sctype) in zip(names(wide_train), sch.scitypes)
    if col == :TARGET
        continue
    end
    
    if sctype <: Textual
        n_levels = length(levels(wide_train[!, col]))
        if n_levels <= 10
            push!(low_card_textual, col)
        else
            push!(high_card_textual, col)
        end
    elseif sctype == Union{Count, Textual}
        push!(mixed_textual, col)
    else
        # Count, Continuous, Union{Continuous,Count} 都算数值列
        push!(numeric_cols, col)
    end
end

# 4. 要剥离的高基数 + mixed 文本列
cols_to_remove = vcat(high_card_textual, mixed_textual)
clean_train = DataFrames.select(wide_train, Not(cols_to_remove))

println("剥离了 $(length(cols_to_remove)) 列高基数/混合文本列：")
println(cols_to_remove)
println("剩余列数：", ncol(clean_train))

# 5. 对低基数文本列做 coerce → Multiclass（后面会自动 onehot）

using Base.Threads

@threads for col in low_card_textual
    coerce!(clean_train, Symbol(col) => Multiclass)
end





using MLJModels, StatsBase



# ============================== 第一步：低基数列彻底 One-Hot（100% 稳）==============================

if !isempty(low_card_textual)
    println("正在进行低基数列 One-Hot 编码（终极兼容版）...")
    
    # 方法1：使用 MLJ 的 pipeline 方式（推荐）
    hot = MLJ.OneHotEncoder(features=Symbol.(low_card_textual), drop_last=false)
    
    # 创建 machine 并训练
    mach = machine(hot, clean_train)
    MLJ.fit!(mach)
    
    # 转换数据
    clean_train = MLJ.transform(mach, clean_train)
    
    println("One-Hot 编码完成！新增了 $(ncol(clean_train) - 127) 个 0/1 列")
    
    # 方法2：或者使用 MLJModels 的直接转换（如果只需要转换不需要保存模型）
    # hot_model = OneHotEncoder(; drop_last=false)
    # hot_transformer = fit(hot_model, clean_train)  # 这个方法来自 MLJModels
    # clean_train = transform(hot_transformer, clean_train)
    
else
    println("没有低基数列需要处理")
end




# ================================== 第二步：所有非 TARGET 列强制转 Continuous ==================================
println("\n开始把所有非 TARGET 列强制转为 Continuous（包含新生成的 0/1 列）...")

feature_cols = setdiff(names(clean_train),  ["TARGET"]) 

@threads  for col in feature_cols
    vec = clean_train[!, col]
    
    # 无论原来是什么鬼东西（Categorical、Real、Any、Missing），全部强转 Float64
    clean_train[!, col] = float.(coalesce.(vec, missing))
    
    # 再安全 coerce
    coerce!(clean_train, Symbol(col) => ScientificTypes.Continuous)
end

println("全部特征列已成功转为 Continuous！")

# ================================== 最终检查 + 准备建模 ==================================
println("\n最终数据状态（黄金状态）：")
schema(clean_train)


y, X = unpack(clean_train,col -> col == :TARGET, col -> col != :TARGET)

@info "数据清洗 100% 完成！现在可以随便扔任何模型！"
println("最终特征维度: $(size(X))")
println("X 的 scitype: $(scitype(X))")   # 应该是 Table{AbstractVector{Continuous}}


# 获取所有可用模型
ms = models(matching(X,y))
# 可直接用的模型数量（通常 300+）
n = ms |> length
println("当前可用模型数量：$n 个")




# 导入必要的模块
using Dates, Serialization

# 现在可以使用 now()
data = (y=y, X=X, metadata=Dict("created"=>now()))
serialize("data/xy", data)