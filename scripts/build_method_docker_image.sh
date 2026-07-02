#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

# method name and destination directory
method_arg="${1:-comp1}"
method_name="${method_arg#cf_}"
function_name="cf_${method_name}"
destination_dir="${project_root}/method_docker"

# path of source and definition file
source_parse_file="${project_root}/cafe/method/function/parse_args.py"
source_decorator_file="${project_root}/cafe/method/function/method_decorator.py"
source_preprocess_file="${project_root}/cafe/method/function/preprocess_pipeline.py"
source_method_file="${project_root}/cafe/method/function/${function_name}.py"
if [[ ! -f "${source_method_file}" ]]; then
	source_method_file="${project_root}/cafe/method/function/${method_name}.py"
fi
source_dockerfile="${project_root}/cafe/method/Dockerfile/${method_name}.dockerfile"
source_pip_requirement="${project_root}/cafe/method/requirement/${method_name}.txt"

destination_parse_file="${destination_dir}/run.py"
destination_decorator_file="${destination_dir}/method_decorator.py"
destination_preprocess_file="${destination_dir}/preprocess_pipeline.py"
destination_method_file="${destination_dir}/${function_name}.py"
destination_dockerfile="${destination_dir}/Dockerfile"
destination_pip_requirement="${destination_dir}/requirement.${method_name}.txt"

mkdir -p "${destination_dir}"

# copy and rename related file
cp "${source_parse_file}" "${destination_parse_file}"
cp "${source_decorator_file}" "${destination_decorator_file}"
cp "${source_preprocess_file}" "${destination_preprocess_file}"
cp "${source_method_file}" "${destination_method_file}"
cp "${source_dockerfile}" "${destination_dockerfile}"
source_method_basename="$(basename "${source_method_file}")"
if [[ "${source_method_basename}" != "${function_name}.py" ]]; then
	sed -i "s/${source_method_basename}/${function_name}.py/g" "${destination_dockerfile}"
fi
if [[ -f "${source_pip_requirement}" ]]; then
	cp "${source_pip_requirement}" "${destination_pip_requirement}"
else
	: >"${destination_pip_requirement}"
fi
echo "Files have been copied and renamed successfully."

# # extract version
# version=$(grep -oP '(?<=version: ).*' "${destination_definition}")
# echo "Extracted version: ${version}"

# build, test and push docker image
# image_name="huangzhaoyang/${method_name}:${version}" # version update
image_name="huangzhaoyang/${method_name}:0.0.1"
docker build -t "${image_name}" "${destination_dir}"
echo "Docker image:${image_name} built successfully."

docker run \
	--rm \
	-v "${project_root}/method_docker_input:/data" \
	--workdir /code \
	--gpus all \
	"${image_name}" \
	python /code/run.py --function_name "${method_name}" --adata_path /data/adata.h5ad --parameters "/data/parameters_${method_name}.json" --output_filename ./output.pkl
echo "Docker image:${image_name} test successfully."

# push docker image on github workflow
# echo "Docker image:${image_name} push successfully."

# for debug: 

# run command mannually
# method_name="dynamo"
# image_name="huangzhaoyang/${method_name}:0.0.1"
# docker run \
# 	--rm \
# 	-v ./method_docker_input:/data \
# 	--workdir /code \
# 	${image_name} \
# 	python /code/run.py --function_name ${method_name} --adata_path /data/adata.h5ad --parameters /data/parameters_${method_name}.json --output_filename ./output.pkl --save_h5ad ./output.h5ad

# enter into the docker image to check file
# docker run \
# 	-it \
# 	-v ./method_docker_input:/data \
# 	--workdir /code \
# 	--entrypoint bash \
# 	${image_name}
