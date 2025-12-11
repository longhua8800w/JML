# 在全新的环境（或临时环境）中执行
using Pkg

# 第一阶段：仅安装绝对必要的包
essential_packages = [
    "MLJ",
    "MLJBase", 
    "MLJTuning",
    "MLJModels",
    "MLJDecisionTreeInterface",
    "DecisionTree",   # 提供稳定树模型
    "DataFrames",     # 数据处理
    "CSV",            # 读写CSV
    "Serialization",   # 读写Julia二进制文件
    "Distributions",
    "Turing",
    "Optim",
    "CategoricalArrays",
    "EvoTrees",
    "MLJScikitLearnInterface",
    "CatBoost",
    "LightGBM",
    "SIRUS",
    "MultivariateStats",
    "BetaML",
    "LIBSVM",
    "NearestNeighborModels",
    "Maxnet"


]

println("安装核心包...")
for pkg in essential_packages
    println(" 安装 $pkg")
    Pkg.add(pkg)
end

# 验证安装
println("\n验证安装...")
for pkg in essential_packages
    try
        eval(Meta.parse("using $pkg"))
        println("✅ $pkg 加载成功")
    catch e
        println("❌ $pkg 加载失败: $e")
    end
end