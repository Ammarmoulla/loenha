script_path=$(readlink -f "$0")
script_dir=$(dirname "$script_path")
unzip /content/drive/MyDrive/models/extra_train.zip
mkdir -p data
tail -n 20000 extra_train.txt > dataset.txt
split -l 18000 dataset.txt
mv xaa train.txt
mv xab valid.txt
mv train.txt valid.txt dataset.txt ${script_dir}/data