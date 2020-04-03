export CUDA_VISIBLE_DEVICE=0
python train_oscc.py \
--n_class 4 \
--data_path "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_sf/subslide/" \
--meta_path "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_sf/trainval_mini.csv" \
--model_path "results/saved_models/" \
--log_path "results/logs/" \
--task_name "fpn_sf_global" \
--mode 1 \
--batch_size 8 \
