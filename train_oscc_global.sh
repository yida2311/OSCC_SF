export CUDA_VISIBLE_DEVICE=0
python train_oscc.py \
-n_class 4 \
--data_path  \
--meta_path  \
--model_path  \
--log_path  \
--task_name "fpn_sf_global" \
--mode 1 \
--batch_size 8 \
