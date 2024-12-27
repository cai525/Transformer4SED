root_path="ROOT-PATH"
save_folder="${root_path}/exps/pmam/run"
config_folder="${root_path}/config/pmam"

mkdir $save_folder

dir1="${save_folder}/pretrain"
dir2="${save_folder}/finetune1"
dir3="${save_folder}/finetune2"

CUDA_NUM=0,1,2
export CUDA_VISIBLE_DEVICES=$CUDA_NUM


source ${root_path}/scripts/mem_check.sh 10000  # check GPU memory

# post-pretrain
cd ${root_path}/recipes/desed/pmam
mkdir $dir1
python main.py --multigpu=True --random_seed=True \
    --config_dir="${config_folder}/post_pretrain.yaml" \
    --save_folder=$dir1 \
    --gmm_means_path="${save_folder}/tokenizer/gmm_means.pt"

sleep 60

cd ${root_path}/recipes/desed/finetune/cnn_trans

# finetune1
source ${root_path}/scripts/mem_check.sh 10000
mkdir $dir2
cp "$dir1/best_model.pt" "$dir2/best_student.pt"  
python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune1.yaml" --save_folder=$dir2
sleep 60

# finetune2
source ${root_path}/scripts/mem_check.sh 20000
source ${root_path}/scripts/process_check.sh  main.py
mkdir $dir3
cp "$dir2/best_student.pt" "$dir3/best_student.pt"
cp "$dir2/best_teacher.pt" "$dir3/best_teacher.pt"
python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune2.yaml" --save_folder=$dir3