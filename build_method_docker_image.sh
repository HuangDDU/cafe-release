#!/bin/bash

# 定义变量
method_name="${1:-cf_paga}"
destination_dir="method_docker"

# 定义源文件和目标文件路径
source_parse_file="cfe/method/function/parse_args.py"
source_method_file="cfe/method/function/${method_name}.py"
source_dockerfile="cfe/method/Dockerfile/${method_name}.dockerfile"
source_definition="cfe/method/definition/${method_name}.yml"

destination_parse_file="${destination_dir}/parse_args.py"
destination_method_file="${destination_dir}/run.py"
destination_dockerfile="${destination_dir}/Dockerfile"
destination_definition="${destination_dir}/definition.yml"

# 创建目标目录（如果不存在）
mkdir -p "${destination_dir}"

# 复制并重命名文件
cp "${source_parse_file}" "${destination_parse_file}"
cp "${source_method_file}" "${destination_method_file}"
cp "${source_dockerfile}" "${destination_dockerfile}"
cp "${source_definition}" "${destination_definition}"
echo "Files have been copied and renamed successfully."

# 提取版本号
version=$(grep -oP '(?<=version: ).*' "${destination_definition}")
echo "Extracted version: ${version}"

# 构建,、测试、发布Docker 镜像
image_name="huangzhaoyang/${method_name}:${version}"
docker build -t ${image_name} "./${destination_dir}"
echo "Docker image:${image_name} built successfully."

docker run \
    --rm \
    -v ./method_docker_input:/data \
    --workdir /code \
    ${image_name}\
    python /code/run.py --adata_path /data/adata.h5ad --prior_information /data/prior_information.json --parameters /data/parameters.json --output_filename /data/output.pkl
echo "Docker image:${image_name} test successfully."

# echo "Docker image:${image_name} push successfully."