#!/bin/bash

# method name and destination directory
method_name="${1:-comp1}"
destination_dir="method_docker"

# path of source and definition file
source_parse_file="cfe/method/function/parse_args.py"
source_decorator_file="cfe/method/function/method_decorator.py"
source_preprocess_file="cfe/method/function/preprocess_pipeline.py"
source_method_file="cfe/method/function/cf_${method_name}.py"
source_dockerfile="cfe/method/Dockerfile/${method_name}.dockerfile"

destination_parse_file="${destination_dir}/run.py"
destination_decorator_file="${destination_dir}/method_decorator.py"
destination_preprocess_file="${destination_dir}/preprocess_pipeline.py"
destination_method_file="${destination_dir}/cf_${method_name}.py"
destination_dockerfile="${destination_dir}/Dockerfile"

mkdir -p "${destination_dir}"

# copy and rename related file
cp "${source_parse_file}" "${destination_parse_file}"
cp "${source_decorator_file}" "${destination_decorator_file}"
cp "${source_preprocess_file}" "${destination_preprocess_file}"
cp "${source_method_file}" "${destination_method_file}"
cp "${source_dockerfile}" "${destination_dockerfile}"
echo "Files have been copied and renamed successfully."

# # extract version
# version=$(grep -oP '(?<=version: ).*' "${destination_definition}")
# echo "Extracted version: ${version}"

# build, test and push docker image
# image_name="huangzhaoyang/${method_name}:${version}" # version update
image_name="huangzhaoyang/${method_name}:0.0.1"
docker build -t ${image_name} "./${destination_dir}"
echo "Docker image:${image_name} built successfully."

docker run \
	--rm \
	-v ./method_docker_input:/data \
	--workdir /code \
	${image_name} \
	python /code/run.py --function_name ${method_name} --adata_path /data/adata.h5ad --parameters /data/parameters_${method_name}.json --output_filename ./output.pkl
echo "Docker image:${image_name} test successfully."

# push docker image on github workflow
# echo "Docker image:${image_name} push successfully."

# for debug: 
# enter into the docker image to check file
# docker run \
# 	-it \
# 	-v ./method_docker_input:/data \
# 	--workdir /code \
# 	--entrypoint bash \
# 	huangzhaoyang/unitvelo:0.0.1

# method_name="unitvelo"
# image_name="huangzhaoyang/${method_name}:0.0.1"
# docker run \
# 	--rm \
# 	-v ./method_docker_input:/data \
# 	--workdir /code \
# 	${image_name} \
# 	python /code/run.py --function_name ${method_name} --adata_path /data/adata.h5ad --parameters /data/parameters_${method_name}.json --output_filename ./output.pkl