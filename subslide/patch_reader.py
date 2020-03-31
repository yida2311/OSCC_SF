import os
import numpy as np
import openslide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image, ImageChops
from torchvision.transforms import ToTensor
import  torch.nn.functional as F
import torch
import math
import random

# from store import *

def save_to_png(img, db_location,coord, file_name):
    """PIL Image saver"""
    fname = file_name + "_" + str(coord[0]) + "_" + str(coord[1]) + "_.png"
    img.save(os.path.join(db_location, fname))


def sample_store_patches(file,
                        file_dir,
                        mask_dir,
                        anno_dir,
                        save_dir,
                        target_dir,
                        label_map,
                        storage_type,
                        patch_size,
                        level,
                        overlap,
                        filter_rate,
                        resize_factor,
                        rows_per_iter):
    ''' Sample patches of specified size from .svs file.
        - file             name of whole slide image to sample from
        - file_dir              directory file is located in
        - mask_dir              directory mask file is located in
        - anno_dir              directory annotation file is located in 
        - save_dir              directory patches is stored
        - target_dir            directory patches mask is stored
        - label_map             dictionary mapping string labels to integers
        - storage_type          the patch storage option 
        - patch_size            size of patches
        - overlap               pixels overlap on each side
        - level                 0 is lowest resolution; level_count - 1 is highest
        - filter_rate           rate of tissue
        - rows_per_txn          how many patches to load into memory at once
                     
        Note: patch_size is the dimension of the sampled patches, NOT equivalent to openslide's definition
        of tile_size. This implementation was chosen to allow for more intuitive usage.
    '''
    file_name = file + '.svs'
    mask_name = file + '.png'
    anno_name = file + '.png'
    tile_size = patch_size - 2 * overlap

    db_location = os.path.join(save_dir, file)
    mask_location = os.path.join(target_dir, file)
    if not os.path.exists(db_location):
        os.makedirs(db_location)
    if not os.path.exists(mask_location):
        os.makedirs(mask_location)

    slide = openslide.open_slide(os.path.join(file_dir, file_name))
    mask = np.array(Image.open(os.path.join(mask_dir, mask_name)).convert('1'))
    anno = np.array(Image.open(os.path.join(anno_dir, anno_name)))
    anno = anno * mask + mask
    anno = Image.fromarray(anno)

    tiles = AugmentZoomGenerator(slide,
                                tile_size,
                                overlap,
                                anno_mask=anno,
                                filter_mask=Image.fromarray(mask),
                                filter_rate=filter_rate,
                                is_filter=True
                                )
    level = tiles.level_count - 1
    x_tiles, y_tiles = tiles.level_tiles[level]

    count = 0
    patches, coords, targets = [], [], []
    print("x_tiles({}), y_tiles({})".format(x_tiles, y_tiles))
    for y in range(y_tiles):
        for x in range(x_tiles):
            print(x, y)
            tile_result = tiles.get_tile_filtered(level, (x, y), resize_factor)
            if tile_result is None:
                continue
            filtered_tile, target = tile_result
            # new_tile = np.asarray(filtered_tile, dtype=np.uint8)
            # if np.shape(new_tile)[0] < 5000 or np.shape(new_tile)[1] < 5000:
            #     continue
            if filtered_tile.size[0] < 5000 or filtered_tile.size[1] < 5000:
                continue

            save_to_png(filtered_tile, db_location, [x, y], file)
            save_to_png(target, mask_location, [x, y], file)
            count += 1
            

        # if (y % rows_per_iter == 0 and y != 0) or y == y_tiles-1:
        #     if storage_type =='png':
        #         db_location = os.path.join(save_dir, file)
        #         mask_location = os.path.join(target_dir, file)
        #         if not os.path.exists(db_location):
        #             os.makedirs(db_location)
        #         if not os.path.exists(mask_location):
        #             os.makedirs(mask_location)
        #         save_to_png(db_location, patches, coords, file)
        #         save_to_png(mask_location, targets, coords, file)
        #     elif storage_type != 'hdf5':
        #         patches, coords, labels = [], [], [] # Reset right away.
        # if storage_type == 'hdf5':
        #     save_to_hdf5(save_dir, patches, coords, file, is_csv=True)
        #     save_to_hdf5(target_dir, targets, coords, file, is_csv=False)
    return count


class  AugmentZoomGenerator(DeepZoomGenerator):
    def __init__(self, osr, tile_size, overlap,  anno_mask=None, filter_mask=None, filter_rate=0.1, is_filter=True, limit_bounds=False,):
        super(AugmentZoomGenerator, self).__init__(osr, tile_size, overlap, limit_bounds)
        self.is_filter = is_filter
        self.filter_mask = filter_mask
        self.anno_mask = anno_mask
        self._osr = osr
        self.scale_factor = 16
        self.filter_rate = filter_rate
    
    def get_tile_filtered(self, level, address, resize_factor):
        """Return an RGB PIL.Image for a tile.
        
        level: the Deep Zoom leel
        address: the address of the tile within the level as a (col, row) tuple
        """
        args, z_size = self._get_tile_info(level, address)
        location = args[0]
        l_size = args[2]
        tile = self._osr.read_region(*args)

        if self.is_filter and self.filter_mask is not None:
            s_location, l_location = self._reset_location(location, l_size)
            cut_region = self.filter_mask.crop(s_location+l_location)
            tissue_mask = np.asarray(cut_region, dtype=np.bool).astype('uint8')
            if self._calculate_mask_rate(tissue_mask) < self.filter_rate:
                return None
        
        anno_mask = self.anno_mask.crop(s_location+l_location)
        anno_mask = anno_mask.resize(tile.size)

        # Apply on solid background
        bg = Image.new('RGB', tile.size, self._bg_color)
        tile = Image.composite(tile, bg, tile)
        if tile.size != z_size:
            tile.thumbnail(z_size, Image.ANTIALIAS)
        tile = tile.resize((int(tile.size[0]/resize_factor), int(tile.size[1]/resize_factor)))
        anno_mask = anno_mask.resize(tile.size)
        
        return tile, anno_mask

    def _reset_location(self, location, size):
        scale_factor = self.scale_factor
        s_location_x = round(location[0] / scale_factor)
        s_location_y = round(location[1] / scale_factor)
        l_location_x = round((location[0]+size[0])/scale_factor)
        l_location_y = round((location[1]+size[1])/scale_factor)
        return (s_location_x, s_location_y),(l_location_x, l_location_y)
    
    def _calculate_mask_rate(self, mask):
        """
        mask: np array
        """
        return mask.sum() / mask.size


def sample_store_boundary_patches(file, 
                                file_dir,
                                mask_dir,
                                save_dir,
                                target_dir,
                                patch_size,
                                level,
                                filter_rate,
                                resize_factor):
    ''' Sample patches of specified size from .svs file around the boundary between normal, mucosa and tumor.
        - file             name of whole slide image to sample from
        - mask_dir              directory mask file is located in
        - save_dir              directory patches is stored
        - target_dir            directory patches mask is stored
        - label_map             dictionary mapping string labels to integers
        - storage_type          the patch storage option 
        - patch_size            size of patches
        - level                 0 is lowest resolution; level_count - 1 is highest
        - filter_rate           rate of tissue
                     
        Note: patch_size is the dimension of the sampled patches, NOT equivalent to openslide's definition
        of tile_size. This implementation was chosen to allow for more intuitive usage.
    '''
    file_name = file + '.svs'
    mask_name = file + '.png'
    base_save_dir = os.path.join(save_dir, file)
    base_target_dir = os.path.join(target_dir, file)

    if not os.path.exists(base_save_dir):
        os.makedirs(base_save_dir)
    if not os.path.exists(base_target_dir):
        os.makedirs(base_target_dir)

    print(mask_dir, mask_name)
    
    slide = openslide.open_slide(os.path.join(file_dir, file_name))
    mask_pil = Image.open(os.path.join(mask_dir, mask_name))
    mask = np.array(mask_pil)
    boundary = get_boundary(mask)  # 计算边界
    bd_row, bd_col = boundary.nonzero()        # 获取非零点（边界点）的坐标
    interval = int(patch_size / 16 / 1.5) # 采样间隔
    num_patch = int(bd_row.size / interval)  # patch数量
    print('num of patches: {}'.format(num_patch))

    patch_generator = PatchGenerator(slide, mask_pil, patch_size)
    points = patch_generator.select_boundary_point(bd_row, bd_col, num_patch, interval)
    num = 0
    for point in points:
        print(point, '>=', num)
        result = patch_generator.get_patch_filtered(point, filter_rate, level)
        if result is not None:
            patch_img, patch_mask = result
            name = file + '_' + str(point[0]) + '_' + str(point[1]) + '.png'
            patch_img.save(os.path.join(base_save_dir, name))
            patch_mask.save(os.path.join(base_target_dir, name))
            num += 1
    return num




def get_boundary(img):
    """获取图像所需的边界
    Params:
        img: numpy array
    """
    img = ToTensor()(img)
    img_b = F.max_pool2d(img, kernel_size=3, stride=1, padding=1)
    bd = img_b - img
    bd_bool = (bd > 0).clone() * (img > 0).clone()
    return bd_bool.numpy().squeeze()

    
class PatchGenerator(object):
    def __init__(self, slide, mask, patch_size, scale=16, resize_factor=2, is_filter=True):
        self.slide = slide
        self.mask = mask 
        self.mask_size = mask.size  # w, h
        self.patch_size = patch_size
        self.scale = scale
        self.patch_size_resize = self.patch_size // self.scale
        self.resize_factor = resize_factor
        self.is_filter = is_filter
    
    def select_boundary_point(self, bd_row, bd_col, num, interval):
        points = []
        dev = np.random.randint(interval, size=num)
        head = np.arange(0, num*interval, interval)
        index = head + dev 
        points_row = bd_row[index]
        points_col = bd_col[index]

        points_row = np.clip(points_row, a_min=self.patch_size_resize//2, a_max=self.mask_size[1]-self.patch_size_resize//2)
        points_col = np.clip(points_col, a_min=self.patch_size_resize//2, a_max=self.mask_size[0]-self.patch_size_resize//2)

        for i in range(num):
            points.append((points_row[i], points_col[i]))
        return points

    def get_patch_filtered(self, pt, filter_rate, level):
        top = pt[0] - self.patch_size_resize // 2
        left = pt[1] - self.patch_size_resize // 2

        patch_mask = self.mask.crop((left, top, left+self.patch_size_resize, top+self.patch_size_resize))

        if self.is_filter and self._calculate_mask_rate(patch_mask) < filter_rate:
                return None
        else:
            top_slide = top * self.scale
            left_slide = left * self.scale
            patch = self.slide.read_region((left_slide, top_slide), level, (self.patch_size, self.patch_size))
            patch = patch.resize((self.patch_size//2, self.patch_size//2))
            return patch, patch_mask
        
    

    def _calculate_mask_rate(self, mask):
        """
        mask: np array
        """
        mask_bin = np.array(mask) > 0
        return mask_bin.sum() / mask_bin.size





11