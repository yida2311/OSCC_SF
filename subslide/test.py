import os
import json

from subslide import Cutter

json_path = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_gl/trainval_test_slide.json'
file_dir = '/media/ldy/7E1CA94545711AE6/OSCC/coarse-key/orig_data/'
mask_dir = '/media/ldy/7E1CA94545711AE6/OSCC/coarse-key/seg/filtered_mask/'
anno_dir = '/media/ldy/7E1CA94545711AE6/OSCC/coarse-key/seg/label_mask/'
save_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_seg/subslide/train/'
target_dir = '/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_seg/subslide/target_train/'
label_map = dict(bgd=0,
                normal=1,
                mucosa=2, 
                tumor=3)
# storage_type = 'png'
with open(json_path, 'r') as f:
    slide_info = json.load(f)

# run_file = os.listdir('/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_seg/subslide/target_train/')
slide_list = slide_info['train']
# slide_list = os.listdir('/media/ldy/e5a10f4e-18fd-4656-80d8-055bc4078655/OSCC_seg/subslide/train/')
# print(len(slide_list))
# print(len(run_file))
# slide_list = [c for c in slide_list if not c in run_file]
# slide_list = ['_20190719181501', '_20190718200940', '_20190403083921', '_20190403101949']
print(slide_list)

cutter = Cutter(slide_list,
                file_dir,
                mask_dir,
                anno_dir,
                save_dir,
                target_dir,
                label_map,
                storage_type)

patch_size = 14000
level = 0
overlap = 2000
filter_rate = 0.1
rows_per_iter = 1
resize_factor = 2
cutter.sample_and_store_patches(patch_size=patch_size,
                                level=level,
                                overlap=overlap,
                                filter_rate=filter_rate,
                                rows_per_iter=rows_per_iter)
