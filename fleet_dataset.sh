script_path=$(readlink -f "$0")
script_dir=$(dirname "$script_path")
unzip /content/shakkelha/dataset/extra_train.zip
mkdir -p data
tail -n 10000 extra_train.txt > dataset.txt
split -l 9000 dataset.txt
mv xaa train.txt
mv xab valid.txt
mv train.txt valid.txt dataset.txt ${script_dir}/data