root_path="/home/cpf/code/open/Transformer4SED"
save_folder="$root_path/exps/mat-sed/base/run"
config_folder="$root_path/config/mat-sed/base"

CUDA_NUM=0,1,2
export CUDA_VISIBLE_DEVICES=$CUDA_NUM
dir1="${save_folder}/pretrain"
dir2="${save_folder}/finetune1"
dir3="${save_folder}/finetune2"

# check GPU memory
source $root_path/scripts/mem_check.sh 10000

# # pretrain
# cd $root_path/recipes/mlm/mlm_passt
# mkdir $dir1
# python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/pretrain.yaml" --save_folder=$dir1
# sleep 60

cd $root_path/recipes/finetune/passt

# finetune1
source $root_path/scripts/mem_check.sh 10000
# mkdir $dir2
# cp "$dir1/best_student.pt" "$dir2/best_student.pt"  
# python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune1.yaml" --save_folder=$dir2
# sleep 60

# finetune2
source $root_path/scripts/process_check.sh  main.py
source $root_path/scripts/mem_check.sh 20000
mkdir $dir3
cp "$dir2/best_student.pt" "$dir3/best_student.pt"
cp "$dir2/best_teacher.pt" "$dir3/best_teacher.pt"
python main.py --multigpu=True --random_seed=True --config_dir="${config_folder}/finetune2.yaml" --save_folder=$dir3
