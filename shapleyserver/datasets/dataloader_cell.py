import cv2
import os
import random as r
import numpy as np

import torch
import torch.utils.data as data

from .. opts import opt

#{{{
def read_files(data_dir, file_name={}):

    image_name = os.path.join(data_dir, 'image', file_name['image'])
    print(data_dir, 'image', file_name['image'])
    exit()
    trimap_name = os.path.join(data_dir, 'trimap', file_name['trimap'])
    alpha_name = os.path.join(data_dir, 'alpha', file_name['alpha'])

    image = cv2.imread(image_name)
    trimap = cv2.imread(trimap_name)
    alpha = cv2.imread(alpha_name)

    return image, trimap, alpha

def random_scale_and_creat_patch(image, patch_size):
    # random scale
    if r.random() < 0.5:
        h, w, c = image.shape
        scale = 0.75 + 0.5*r.random()
        image = cv2.resize(image, (int(w*scale),int(h*scale)), interpolation=cv2.INTER_CUBIC)
    # creat patch
    if r.random() < 0.5:
        h, w, c = image.shape
        if h > patch_size and w > patch_size:
            x = r.randrange(0, w - patch_size)
            y = r.randrange(0, h - patch_size)
            image = image[y:y + patch_size, x:x+patch_size, :]
        else:
            image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)
    else:
        image = cv2.resize(image, (patch_size,patch_size), interpolation=cv2.INTER_CUBIC)

    return image
#}}}

class XrayDataLoader(data.Dataset):
    """
    human_matting
    """

    def __init__(self, root_dir, mode, patch_size, sub_dir = None):
        super().__init__()
        assert mode in ['train', 'test']

        self.root = root_dir
        self.mode = mode
        self.patch_size = 224 #patch_size
        self.sub_dir = sub_dir

        #import pdb;pdb.set_trace()
        self.imgID_normal, self.imgID_cnv, self.imgID_dme, self.imgID_drusen = self.make_xray_dataset()
        self.num_normal, self.num_cnv = len(self.imgID_normal), len(self.imgID_cnv)
        self.num_dme, self.num_drusen = len(self.imgID_dme), len(self.imgID_drusen)
        self.index_normal, self.index_pneu = 0, 0
        self.index_dme, self.index_drusen = 0, 0

        self.imgID_all = self.imgID_normal + self.imgID_cnv + self.imgID_dme + self.imgID_drusen
        self.num = len(self.imgID_all)
        print("Dataset %s : file number %d" % (root_dir, self.num))

    def __len__(self):
        if self.mode == 'train':
            #if opt.is_defense:
            #    return int(self.num/2)
            return self.num
        else:
            return self.num

    def __getitem__(self, index):
        # read files
        #if index % 2 == 0:

        if self.mode == 'train':
            if index >= self.num:
                raise IndexError()

            image_name, label = self.imgID_all[index]
            '''
            r_id = r.random()
            if r_id < 0.25:
                image_name, label = self.imgID_normal[index % self.num_normal]
            elif r_id < 0.5:
                image_name, label = self.imgID_cnv[index % self.num_cnv]
            elif r_id < 0.75:
                image_name, label = self.imgID_dme[index % self.num_dme]
            else:
                image_name, label = self.imgID_drusen[index % self.num_drusen]
            '''
        else:
            #print(self.index_normal, self.num_normal, self.index_pneu, self.num_pneu)
            image_name, label = self.imgID_all[index]

        image = cv2.imread(image_name)

        # augmentation
        #print('before', image.shape)
        image = self.scale_image(image)
        #print('scale', image.shape)
        if self.mode == 'train':
            image = self.random_flip(image)

        # normalize
        image = image.astype(np.float32) / 255.0
        #image = (image.astype(np.float32)  - (114., 121., 134.,)) / 255.0

        # to tensor
        image = self.np2Tensor(image)

        #h, w, c = image.shape
        #image = torch.from_numpy(image.transpose((2, 0, 1))).view(c, h, w).float()
        #label = torch.from_numpy(label.astype(np.int32))
        return {'image': image, 'label': label, 'image_name': image_name }

    def read_img(self, class_name, label):
        items = []
        file_path = os.path.join(self.root, self.mode if self.sub_dir == None else self.sub_dir, class_name)
        images_list = os.listdir(file_path)
        for image_path in images_list:
            items.append([os.path.join(file_path, image_path), label])
        return items

    def make_xray_dataset(self):
        items_normal = self.read_img('NORMAL', 0)
        items_cnv    = self.read_img('CNV',    1)
        items_dme    = self.read_img('DME',    2)
        items_drusen = self.read_img('DRUSEN', 3)

        return items_normal, items_cnv, items_dme, items_drusen

    def scale_image(self, image):
        image = cv2.resize(image, (self.patch_size, self.patch_size), interpolation=cv2.INTER_CUBIC)
        return image

    def random_flip(self, image):

        if r.random() < 0.5:
            image = cv2.flip(image, 0)

        if r.random() < 0.5:
            image = cv2.flip(image, 1)
        return image

    def np2Tensor(self, array):
        ts = (2, 0, 1)
        tensor = torch.FloatTensor(array.transpose(ts).astype(float))
        return tensor

