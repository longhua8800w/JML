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
transform(_, names(_) .=> ByRow(x -> ismissing(x) ? 0 : x) .=> names(_))
# 处理缺失值（示例：填充为0）


# 对test做相同操作（无TARGET）
wide_test = @pipe test |>
leftjoin(_, bureau_agg, on=:SK_ID_CURR) |>  
leftjoin(_, prev_app_agg, on=:SK_ID_CURR) |>
transform(_, names(_) .=> ByRow(x -> ismissing(x) ? 0 : x) .=> names(_))
# 处理缺失值（示例：填充为0）


using MLJ
using ScientificTypes

scm_tb = schema(wide_train)

first(wide_train, 4) |> pretty
f(x) = Textual <: x 
map(f, scm_tb.scitypes)

f.(scm_tb.scitypes)

map( x->Textual <: x , scm_tb.scitypes)

using DataFrames
not_textual_colnms = @pipe scm_tb|>
 DataFrame|> 
 DataFrames.transform( _,
    :scitypes => ByRow( x->Textual <: x ) => :is_textual) |> 
    subset(_,  :is_textual => x -> x.!= true) |>
_.names 


@pipe wide_train|> select(_, :TARGET) 



train = @pipe wide_train |> select(_,not_textual_colnms) |> coerce(_, :TARGET => Multiclass)

dropmissing!(train)

schema(train)


y, X = @pipe unpack(train, ==(:TARGET), !=(:SK_ID_CURR), rng=123) 


scitype(y)
scitype(X)

(Xtrain, Xtest), (ytrain, ytest) = partition((X, y), 0.6, multi=true, rng=123)

matching(X,y)

ms = models(matching(X,y))



using MLJScientificTypes:names
names(X, Count)

using ScientificTypes
import ScientificTypes: scitype

scitype.(eachcol(X))

names(X, eltype.(ScientificTypes.scitype.(eachcol(X)) ) .>: Count)

X1 = MLJ.coerce(X, Symbol.(names(X, eltype.(scitype.(eachcol(X)) ) .>: Count)) .=> ScientificTypes.Continuous )


# 选择科学类型为 Count 或子类型的列
count_cols = names(X, eltype.(ScientificTypes.scitype.(eachcol(X)) ) .>: Count)

# 将这些列转换为 Continuous 类型
# 将生成器转换为元组
coercion_specs = Tuple( Symbol(col) => Continuous for col in count_cols)

X_continuous = ScientificTypes.coerce(X, coercion_specs...)


names(X,eltype.(eachcol(X)) .== Real)



coerce_spec = Tuple( Symbol(col) => Continuous for col in names(X,eltype.(eachcol(X)) .== Real))


X_continuous  = @pipe X |>
    DataFrames.transform(_, All() .=> (x -> convert.(Float64, x)) .=> identity) |>  # 转换为Float64  抽象类型后面会报错
    MLJ.coerce(_, coerce_spec...)




ms = models(matching(X_continuous,y))


models() do model
    matching(model, X_continuous, y) &&
    model.prediction_type == :deterministic &&
    model.is_pure_julia
end;

