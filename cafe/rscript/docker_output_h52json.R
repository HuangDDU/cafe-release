#!/usr/bin/Rscript

####################################################### 参数提取部分#######################################################
# 从命令行获取json和h5文件名
library(optparse)
option_list <- list(
    make_option(c("--output_h5_filename"), type = "character", help = "Output h5 filename from Docker"),
    make_option(c("--output_json_filename"), type = "character", help = "Output json filename for Python")
)
opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
output_h5_filename <- opt$output_h5_filename
output_json_filename <- opt$output_json_filename

####################################################### h5 -> json#######################################################
library(jsonlite)

output <- dynutils::read_h5(output_h5_filename)
output <- toJSON(output)
write(output, f = output_json_filename)

"h52json successful!"
