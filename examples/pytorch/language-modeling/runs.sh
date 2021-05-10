python run_clm.py --config_name configs/gpt2_small.json --model_type gpt2 --tokenizer_name gpt2 --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --do_train --do_eval --output_dir ./local_gpt2_testing --per_device_train_batch_size 36 --dataloader_num_workers 8 --preprocessing_num_workers 8 --overwrite_output_dir --block_size 512



# NUM_GPUS=$1
# CONFIG=$2
# SAMPLES_PER_GPU=$3
# BATCH_SIZE=$4
# LR=$5
# WEIGHT_DECAY=$6
# MAX_GRAD_NORM=$7
# NUM_STEPS=$8
# SUFFIX=$9

# shift 9

# SAMPLES_PER_STEP=$((SAMPLES_PER_GPU * NUM_GPUS))

# export PYTHONPATH=$PYTHONPATH:/cb/home/ryanh/ws/huggingface-gpt/py_root
# source /cb/home/ryanh/ws/fairseq_kernel/bin/activate

# python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS run_clm.py --config_name configs/$CONFIG.json --tokenizer_name gpt2 --dataset_name wikitext --dataset_cache_dir /cb/ml/language/datasets/huggingface/ --dataset_config_name wikitext-2-raw-v1 --dataset_intmd_dir /cb/ml/language/datasets/huggingface/wikitext/cache_dir --dataset_split train[:-5000] train[-5000:] --per_device_train_batch_size $SAMPLES_PER_GPU --per_device_eval_batch_size $SAMPLES_PER_GPU --gradient_accumulation_steps $((BATCH_SIZE / SAMPLES_PER_STEP)) --evaluation_strategy steps --do_train --do_eval --output_dir /cb/home/ryanh/extra-storage/huggingface/wikitext/${CONFIG}_${NUM_GPUS}gpu_${BATCH_SIZE}bs_${LR}lr_${WEIGHT_DECAY}wd_${MAX_GRAD_NORM}mgn_${SUFFIX}_${NUM_STEPS} --learning_rate $LR --weight_decay $WEIGHT_DECAY --max_grad_norm $MAX_GRAD_NORM --max_steps $NUM_STEPS --logging_dir runs --fp16 --save_steps 1000 --eval_steps 1000 --evaluation_strategy steps --dataloader_num_workers 8 --preprocessing_num_workers 8 --overwrite_output_dir --logging_steps 10 $@
