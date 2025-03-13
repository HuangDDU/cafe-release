#!/bin/bash

# method name and destination directory
method_name="${1:-cf_paga}"
destination_dir="method_docker"

# path of source and definition file
source_parse_file="cfe/method/function/parse_args.py"
source_method_file="cfe/method/function/${method_name}.py"
source_dockerfile="cfe/method/Dockerfile/${method_name}.dockerfile"
source_definition="cfe/method/definition/${method_name}.yml"

destination_parse_file="${destination_dir}/parse_args.py"
destination_method_file="${destination_dir}/run.py"
destination_dockerfile="${destination_dir}/Dockerfile"
destination_definition="${destination_dir}/definition.yml"

mkdir -p "${destination_dir}"

# copy and rename related file
cp "${source_parse_file}" "${destination_parse_file}"
cp "${source_method_file}" "${destination_method_file}"
cp "${source_dockerfile}" "${destination_dockerfile}"
cp "${source_definition}" "${destination_definition}"
echo "Files have been copied and renamed successfully."

# extract version
version=$(grep -oP '(?<=version: ).*' "${destination_definition}")
echo "Extracted version: ${version}"

# build, test and push docker image
image_name="huangzhaoyang/${method_name}:${version}"
docker build -t ${image_name} "./${destination_dir}"
echo "Docker image:${image_name} built successfully."

docker run \
	--rm \
	-v ./method_docker_input:/data \
	--workdir /code \
	${image_name} \
	python /code/run.py --adata_path /data/adata.h5ad --prior_information /data/prior_information.json --parameters /data/parameters.json --output_filename /data/output.pkl
echo "Docker image:${image_name} test successfully."

# push docker image on github workflow
# echo "Docker image:${image_name} push successfully."
