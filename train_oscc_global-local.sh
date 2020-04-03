export CUDA_VISIBLE_DEVICE=0
python train_oscc.py \
--n_class 4 \
--data_path "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_sf/subslide/" \
--meta_path "/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_sf/trainval_mini.csv" \
--model_path "./results/saved_models/" \
--log_path "./results/logs/" \
--task_name "fpn_sf_global-local" \
--mode 3 \
--batch_size 1 \
--path_g  "fpn_sf_global/fpn_sf_global.epoch2.pth"\
--path_l  "fpn_sf_local/fpn_sf_local.epoch2.pth" \
