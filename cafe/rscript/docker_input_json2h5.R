#!/usr/bin/Rscript

####################################################### 参数提取部分#######################################################
# 从命令行获取json和h5文件名
library(optparse)
option_list <- list(
  make_option(c("--input_json_filename", "-j"), type = "character", help = "Input json filename from Python"),
  make_option(c("--input_h5_filename"), type = "character", help = "Input h5 filename for Docker")
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
input_json_filename <- opt$input_json_filename
input_h5_filename <- opt$input_h5_filename

####################################################### json -> h5#######################################################
library(jsonlite)
library(Matrix)

input <- fromJSON(input_json_filename)

# 这里表达矩阵是稀疏矩阵, 需要单独处理
expression_id <- input$expression_id # 目标表达矩阵expression, counts
expression <- input[[expression_id]]
x <- as.double(expression$x)
i <- expression$i
p <- expression$p
rownames <- as.vector(expression$rownames)
colnames <- as.vector(expression$colnames)
Dim <- expression$Dim
input[[expression_id]] <- new("dgCMatrix", x = x, i = i, p = p, Dim = Dim, Dimnames = list(rownames, colnames))

# 其余的先验知识, 参数, 随机种子等都可以直接传递

dynutils::write_h5(input, input_h5_filename)

"json2h5 successful!"
