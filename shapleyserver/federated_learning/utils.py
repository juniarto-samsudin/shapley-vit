import time
import os
import sys
import copy
import logging
from copy import deepcopy
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import datasets, transforms
from scipy.ndimage.interpolation import rotate as scipyrotate
from sklearn.metrics import confusion_matrix, roc_auc_score, balanced_accuracy_score, accuracy_score
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.decomposition import PCA, KernelPCA
import math
# from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

from shapleyserver.federated_learning.networks import MLP, MLP_tabular, MLP_linear, ConvNet, LeNet, AlexNet, AlexNetBN, VGG11, VGG11BN, ResNet18, ResNet18BN_AP, ResNet18BN, ResNet50
from torch.autograd import Variable

# generic purpose utils
def mkdirs(dirpath):
    try:
        os.makedirs(dirpath)
    except Exception as _:
        pass

def get_logger(logger_path):
    logging.basicConfig(
        filename=logger_path,
        # filename='/home/qinbin/test.log',
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%m-%d %H:%M', 
        level=logging.DEBUG, 
        filemode='w'
    )

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    logger.addHandler(ch)

    return logger 

# data related
def sample_pseudo_img(mean, std, data_info, device):
    """ use this function for initialization with sampled pseudo data from the mean and std calculated from the real data
        args:
            mean, std is the calculated mean and standard deviation of the real images batch, calculated over dim 0
            data_info is a tuple (# samples, # channel, size 0, size 1)
    """
    num_of_samples, num_channel, size_0, size_1 = data_info[0], data_info[1], data_info[2], data_info[3]

    # Generate new pseudo images using the calculated mean and standard deviation
    new_images = torch.zeros((num_of_samples, num_channel, size_0, size_1), device=device)
    for i in range(num_of_samples):
        new_images[i] = torch.randn((num_channel, size_0, size_1), device=device) * (std * 0.2) + mean

    # Clip the pixel values to ensure they are within the range of 0 to 255
    new_images = torch.clamp(new_images, 0, 255)

    # Compute the mean and standard deviation of the generated images
    new_mean = torch.mean(new_images.float(), dim=(0, 2, 3))
    new_std = torch.std(new_images.float(), dim=(0, 2, 3))

    # Normalize the generated images using torchvision's transforms
    transform = transforms.Compose([
        # transforms.ToTensor(),
        transforms.Normalize(new_mean.tolist(), new_std.tolist()),
        # transforms.ToPILImage()
    ])
    new_images_normalized = torch.stack([transform(img) for img in new_images], dim=0)

    return new_images_normalized




class MyIsic(Dataset):
    def __init__(self, img_dir, meta_file, train=True, transform=None):
        # self.root_dir = root_dir
        self.img_dir = img_dir
        self.df = pd.read_csv(meta_file)
        self.df2 = self.df.query("fold == 'train'") if train else self.df.query("fold == 'test'")
        self.targets = self.df2['target'].to_numpy()
        self.img_names = self.df2['image'].to_list()
        self.transform = transform
        # self.img_dir = os.path.join(root_dir, 'ISIC_2019_Training_Input_preprocessed')

    def __getitem__(self, index):
        target = self.df2.iloc[index]['target']
        img_path = os.path.join(self.img_dir, self.img_names[index] +'.jpg')
        image = Image.open(img_path)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    def __len__(self):
        return len(self.df2)

def get_isic(
    img_dir,
    meta_file,
    resized='64x'
):
    if resized == '64x':
        mean = [0.5894, 0.5666, 0.5575]
        std = [0.1984, 0.2135, 0.2188]
    else:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize([64, 64]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    ds_train = MyIsic(img_dir, meta_file, train=True, transform=transform)
    ds_test= MyIsic(img_dir, meta_file, train=False, transform=transform)

    data_info = {}
    data_info['num_classes']=8
    data_info['channel']=3
    data_info['img_size']=(64,64) 
    data_info['mean'] = mean
    data_info['std'] = std
    data_info['train_labels'] = ds_train.targets

    return ds_train, ds_test, data_info

class DrKaggle(Dataset):
    def __init__(self, root_dir, csvfile_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.df = pd.read_csv(csvfile_dir)
        self.targets = self.df['Label'].to_numpy()
        self.img_path = self.df['ImgPath'].to_list()

    def __getitem__(self, index):
        target = self.df.iloc[index]['Label']
        img_path = os.path.join(self.root_dir, self.img_path[index])
        image = Image.open(img_path)
        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)   

        if len(image.split()) != 3:
            image = transforms.Grayscale(num_output_channels=3)(image)

        if self.transform is not None:
            image = self.transform(image)
        
        return image, target
    
    def __len__(self):
        return len(self.df)

def get_dr_dataset(root_path, train_csv, test_csv, val_csv=None):
    # data_base_path = './dr-kaggle/'
    
    img_w, img_h = 256, 256
    data_set, data_info = {}, {}

    mean = [0.3199, 0.2241, 0.1609]
    std = [0.3019, 0.2183, 0.1742]
   
    transform_train = transforms.Compose([
        # transforms.Resize([img_w, img_h]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation((-30, 30)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_test = transforms.Compose([
        # transforms.Resize([img_w, img_h]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    dr_trainset = DrKaggle(root_path, train_csv, transform=transform_train)
    dr_testset = DrKaggle(root_path, test_csv, transform=transform_test)
    # dr_testloader = DataLoader(dataset=dr_testset, batch_size=64, shuffle=False)    

    data_set['train_data']=dr_trainset
    data_set['test_data']= dr_testset
    data_set['train_labels']=dr_trainset.targets
    data_set['test_labels']=dr_testset.targets
    
    data_info['channel'] = 3
    data_info['img_size'] = (img_w, img_h) 
    data_info['num_classes'] = 5
    data_info['mean'] = mean
    data_info['std'] = std

    if val_csv:
        dr_valset = DrKaggle(root_path, val_csv, transform=transform_test)
        data_set['valid_data']=dr_valset

    return data_set, data_info

def get_covid_dataset(root_path):
    mean = [0.4924, 0.4925, 0.4925]
    std = [0.2329, 0.2329, 0.2329]
    normalizer = transforms.Normalize(mean=mean, std=std)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(),
            transforms.ToTensor(),
            normalizer
        ]),
        
        'validation': transforms.Compose([
            transforms.Resize((244, 244)),
            transforms.ToTensor(),
            normalizer
        ])
    }
    
    train_dir = os.path.join(root_path, 'train')
    test_dir = os.path.join(root_path, 'test')

    ds_train = datasets.ImageFolder(train_dir, data_transforms['train'])
    ds_test = datasets.ImageFolder(test_dir, data_transforms['validation'])

    data_info={}
    data_info['train_labels'] = np.array(ds_train.targets, dtype=np.int32)
    data_info['num_classes'] = 3
    data_info['channel'] = 3
    data_info['img_size'] = (244, 244) 
    data_info['mean'] = mean
    data_info['std'] = std

    return ds_train, ds_test, data_info   

def get_dataset(dataset):
    data_set, data_info = {}, {}
    group_valid_dataset = None
    # num_classes=10
    if dataset in ['MNIST', 'mnist']:
        channel = 1
        im_size = (28, 28)
        num_classes = 10
        mean = [0.1307]
        std = [0.3081]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform) # no augmentation
        data_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        train_labels = data_train.targets.numpy()
        test_labels = data_test.targets.numpy()
        mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

    elif dataset in ['EMNIST', 'emnist']:
        channel = 1
        im_size = (28, 28)
        num_classes = 62
        mean = [0.5]
        std = [0.5]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.EMNIST(root="./data", split="byclass", download=True, train=True, transform=transform)
        data_test = datasets.EMNIST(root="./data", split="byclass", download=True, train=False, transform=transform)   
        class_names = [str(c) for c in range(num_classes)]
        train_labels = data_train.targets.numpy()
        test_labels = data_test.targets.numpy()
        mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
        'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
        'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
        'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'], dtype='<U1')

    elif dataset in ['SVHN', 'svhn']:
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.SVHN(root="./data/SVHN", split='train', download=True, transform=transform)  # no augmentation
        data_test = datasets.SVHN(root="./data/SVHN", split='test', download=True, transform=transform)
        class_names = [str(c) for c in range(num_classes)]
        train_labels = data_train.labels
        test_labels = data_test.labels
        mapp = np.array(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], dtype='<U1')

    elif dataset in ['CIFAR10', 'cifar10']:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.CIFAR10(root='./data/CIFAR10', train=True, download=True, transform=transform) # no augmentation
        data_test = datasets.CIFAR10(root='./data/CIFAR10', train=False, download=True, transform=transform)
        print(data_train)



        # """
        # train validation split
        validation_split = .5
        shuffle_dataset = True
        
        test_size = len(data_test)
        indices = list(range(test_size))
        split = int(np.floor(validation_split * test_size))

        if shuffle_dataset :
            random_seed = 42
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        test_indices, val_indices = indices[split:], indices[:split]
        
        from copy import deepcopy
        data_val = deepcopy(data_test)
        data_val.data = data_test.data[val_indices]
        data_val.targets = list(np.array(data_test.targets)[val_indices])

        data_test.data = data_test.data[test_indices]
        data_test.targets = list(np.array(data_test.targets)[test_indices])
        # """


        class_names = data_train.classes
        train_labels = np.array(data_train.targets, dtype=np.int32)
        valid_labels = np.array(data_val.targets, dtype=np.int32)
        test_labels = np.array(data_test.targets, dtype=np.int32)
        mapp = np.array(data_test.classes)

        # print(len(data_train), len(data_val), len(data_test))
        # quit()


    elif dataset in ['CIFAR100', 'cifar100']:
        channel = 3
        im_size = (32, 32)
        num_classes = 100
        mean = [0.5071, 0.4866, 0.4409]
        std = [0.2673, 0.2564, 0.2762]
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        data_train = datasets.CIFAR100(root='./data/CIFAR100', train=True, download=True, transform=transform) # no augmentation
        data_test = datasets.CIFAR100(root='./data/CIFAR100', train=False, download=True, transform=transform)
        class_names = data_train.classes
        train_labels = np.array(data_train.targets, dtype=np.int32)
        test_labels = np.array(data_test.targets, dtype=np.int32)
        mapp = np.array(data_test.classes)


    elif dataset in ['CINIC10', 'cinic10', 'CINIC10-IMAGENET', 'cinic10-imagenet']:
        if ('IMAGENET' in dataset) or ('imagenet' in dataset):
            cinic_directory = 'data/cinic-10-imagenet'
            cinic_train_dir = cinic_directory + '/train'
            cinic_test_dir = cinic_directory + '/test'
        else:
            cinic_directory = 'data/CINIC-10'
            cinic_train_dir = cinic_directory + '/train'
            cinic_test_dir = cinic_directory + '/test'
            cinic_val_dir = cinic_directory + '/valid'
        
        channel = 3
        im_size = (32, 32)
        num_classes = 10
        mean = [0.47889522, 0.47227842, 0.43047404]
        std = [0.24205776, 0.23828046, 0.25874835]
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=mean,std=std)])
        data_train = datasets.ImageFolder(cinic_train_dir, transform)
        data_test = datasets.ImageFolder(cinic_test_dir, transform)
        class_names = data_train.classes
        train_labels = np.array(data_train.targets, dtype=np.int32)
        test_labels = np.array(data_test.targets, dtype=np.int32)
        
        if dataset in ['CINIC10', 'cinic10']:
            data_val = datasets.ImageFolder(cinic_val_dir, transform)
            val_labels = np.array(data_train.targets, dtype=np.int32)
        
        mapp = np.array(data_train.classes)

    elif dataset in ['COMPAS', 'compas', 'adult', 'Adult']:
        import sys
        from sklearn.model_selection import train_test_split
        sys.path.append("../bias-explainer/")
        from data.objects.compas import Compas
        from data.objects.adult import Adult
        from fairxplainer import utils

        class TabularDataset(Dataset):
            def __init__(self, data, target):
                self.data = torch.from_numpy(data.astype(np.float32))
                self.data = self.data.view(-1, 1, 1, self.data.shape[1])
                self.targets = target
                self.n_samples = len(target)
                     

            def __len__(self):
                return self.n_samples

            def __getitem__(self, idx):
                return self.data[idx], self.targets[idx]


        # getting data from FairXplainer
        if(dataset in ['COMPAS', 'compas']):
            data_object = Compas(verbose=False, config=1) # config defines configuration for sensitive groups
        elif(dataset in ['adult', 'Adult']):
            data_object = Adult(verbose=False, config=2)
        else:
            raise ValueError("Unknown dataset")
        df = data_object.get_df()
        X = df.drop(['target'], axis=1)
        y = df['target']
        X = utils.get_one_hot_encoded_df(X, data_object.categorical_attributes)
        print('X:', X.head())
        X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)
        data_train = TabularDataset(X_train.values, y_train.values)
        data_val = TabularDataset(X_val.values, y_val.values)
        data_test = TabularDataset(X_test.values, y_test.values)
        

        # For evaluating group fairness metrics
        group_valid_dataset = []
        for sensitive_feature in data_object.known_sensitive_attributes:
            for value in X_val[sensitive_feature].unique():
                idx = X_val[sensitive_feature] == value
                group_valid_dataset.append(TabularDataset(X_val[idx].values, y_val[idx].values))
        
        
        num_classes = 2
        im_size = X_train.shape[1]
        train_labels = np.array(data_train.targets, dtype=np.int32)
        valid_labels = np.array(data_val.targets, dtype=np.int32)
        test_labels = np.array(data_test.targets, dtype=np.int32)
        class_names = [0, 1]
        mapp = np.array(class_names)
        transform = None
        channel = None
        mean = None
        std = None
        



    else:
        sys.exit('unknown dataset: %s'%dataset)

    data_set['train_data']=data_train
    data_set['test_data']=data_test
    data_set['transform']=transform
    data_set['train_labels']=train_labels
    data_set['test_labels']=test_labels
    data_set['mapp']=mapp
    data_set['group_valid_dataset']=group_valid_dataset

    data_info['channel']=channel
    data_info['img_size']=im_size
    data_info['num_classes']=num_classes
    data_info['class_names']=class_names
    data_info['mean']=mean
    data_info['std']=std
    
    if dataset in ['CINIC10', 'cinic10']:
        data_set['valid_data']=data_val
        data_set['valid_labels']=val_labels

    if(dataset in ['compas', 'COMPAS', 'adult', 'Adult', 'CIFAR10', 'cifar10']):
        data_set['valid_data']=data_val
        data_set['valid_labels']=valid_labels
    # testloader_server = torch.utils.data.DataLoader(data_test, batch_size=256, shuffle=False, num_workers=0)
    
    return data_set, data_info

class CustomSubset(Subset):
    '''A custom subset class with customizable data transformation'''
    def __init__(self, dataset, indices, subset_transform=None):
        super().__init__(dataset, indices)
        self.subset_transform = subset_transform
        
    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        
        if self.subset_transform:
            x = self.subset_transform(x)
      
        return x, y 

def show_data_histogram_client(labels, client_idcs, client_id, mapp):
    plt.figure(figsize=(20,3))
    plt.hist(
        labels[client_idcs], stacked=True, bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),label="Client {}".format(client_id)
        )
    plt.xticks(np.arange(10), mapp)
    plt.legend()
    plt.show()


def partition_labeldir(targets, num_classes=10, n_parties=10, beta=1.0, distributions=None, seed=42):
    """ This data partition function is a copy of that used in paper Federated Learning on Non-IID Data Silos: An Experimental Study.
        dataset: can be both train or test set
        targeets: should be train labels or test labels accordingly
        distributions: if using an existing distributions, pls specify one that generated by np.random.dirichlet()
    """
    min_size = 0
    min_require_size = 10

    # if dataset in ('celeba', 'covtype', 'a9a', 'rcv1', 'SUSY'):
    #     K = 2
    # elif dataset in ['MNIST', 'mnist', 'CIFAR10', 'cifar10', 'SVHN', 'svhn']:
    #     K=10
    # elif dataset in ['cifar100', 'CIFAR100']:
    #     K = 100
    # elif dataset == 'tinyimagenet':
    #     K = 200

    # client distribution should be controlled by seed in Shapley experiments
    np.random.seed(seed)
    
    N = targets.shape[0]
    #np.random.seed(2020)
    net_dataidx_map = {}
    
    if distributions is None:
        distributions = np.random.dirichlet(np.repeat(beta, n_parties), num_classes)

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            np.random.shuffle(idx_k)
            # proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = distributions[k]
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        
    return distributions, net_dataidx_map

def partition_labeldir2(targets, num_classes=10, n_parties=10, beta=1.0, distributions=None, min_class_size=10):
    r""" This function is revised based on `partition_labeldir()`.

    The revision is made to guarantee that each client has at least one class of data with number of samples no less
    than a pre-defined argument `min_class_size` which can be set equal to IPC
    """
    N = targets.shape[0]
    net_dataidx_map = {}

    if distributions is None:
        distributions = np.random.dirichlet(np.repeat(beta, n_parties), num_classes)

    idx_batch = [[] for _ in range(n_parties)]
    for k in range(num_classes):
        idx_k = np.where(targets == k)[0]
        np.random.shuffle(idx_k)
        proportions = distributions[k]
        proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]

        # If proportions result in any batch size less than min_class_size, adjust the proportions
        proportions = np.concatenate([[0], proportions])
        for i in range(1, len(proportions)):
            if proportions[i] - proportions[i-1] < min_class_size and proportions[i] < len(idx_k):
                diff = min(min_class_size - (proportions[i] - proportions[i-1]), len(idx_k) - proportions[i])
                proportions[i:] += diff

        idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions[1:]))]

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]

    return distributions, net_dataidx_map


def partition_labeldir_med(dataset_name, y_train, n_parties, beta=0.1):
    """ This data partition function is a copy of that used in paper Federated Learning on Non-IID Data Silos: An Experimental Study.
        This data partition function is imported as an alternative benchmark
    """
    min_size = 0
    min_require_size = 10

    if dataset_name == 'isic2019':
        K = 8
        # min_require_size = 100
    elif dataset_name =='dr-kaggle':
        K = 5
    elif dataset_name == 'covid-19':
        K = 3
    elif dataset_name in ['organamnist', 'organcmnist', 'organsmnist']:
        K = 11
    elif dataset_name == 'pathmnist':
        K = 9
    elif dataset_name in ['bloodmnist', 'tissuemnist']:
        K = 8
    elif dataset_name == 'dermamnist':
        K = 7
    elif dataset_name == 'octmnist':
        K = 4
    elif dataset_name in ['pneumoniamnist', 'breastmnist']:
        K = 2


    N = y_train.shape[0]
    #np.random.seed(2020)
    net_dataidx_map = {}

    while min_size < min_require_size:
        idx_batch = [[] for _ in range(n_parties)]
        for k in range(K):
            idx_k = np.where(y_train == k)[0]
            np.random.shuffle(idx_k)
            proportions = np.random.dirichlet(np.repeat(beta, n_parties))
            proportions = np.array([p * (len(idx_j) < N / n_parties) for p, idx_j in zip(proportions, idx_batch)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in idx_batch])

    for j in range(n_parties):
        np.random.shuffle(idx_batch[j])
        net_dataidx_map[j] = idx_batch[j]
        
    return net_dataidx_map

def record_net_data_stats(y_train, net_dataidx_map, logger=None):

    net_cls_counts = {}
    if net_dataidx_map is not None:
        for net_i, dataidx in net_dataidx_map.items():
            unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
            tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
            net_cls_counts[net_i] = tmp
            if logger is not None:
                logger.info('Client {:2d} total train data: {:5d}, distribution: {}'.format(net_i, len(dataidx), tmp))
            else:
                print('Client {:2d} total train data: {:5d}, distribution: {}'.format(net_i, len(dataidx), tmp))
    else:
        unq, unq_cnt = np.unique(y_train, return_counts=True)
        for i in range(len(unq)):
            net_cls_counts[unq[i]] = unq_cnt[i]
           
    return net_cls_counts

def make_client_dataset_from_partition(data, num_clients, data_idcs, transform=None):
    client_data = {}
    for client_id in range(num_clients):
        client_data[client_id] = CustomSubset(dataset=data, indices=data_idcs[client_id], subset_transform=transform)
    return client_data


# network/model related utils
def model_sync(src, target_net):
    """ update the weights of target network with that of the source network
        source can be server object or cluster object
        target network can be the cluster.cluster_model or one of the follwoing: 
        client.model, client.global_model_cache, client.cluster_model_cache
    """
    # download the state of the global model or the cluster model
    target_net.load_state_dict(src.model_state)

def copy_parameters(target, source):
    """copy weight parameters from source model to target model
    args: target, source is torch.nn.Parameters.parameters, e.g., model.parameters()
    """
    for target_param, src_param in zip(target, source):
        target_param.data = src_param.data.clone()

def add_net_state(parties, ratio):
    """ implements the computation of model aggregatoin as per input ratio
        parties can be clients or clusters
        ratio is a list contains the coefficients assigned for each network
    """

    w_list = [p.model_state for p in parties]
    w_avg = w_list[0]
    for i, w in enumerate(w_list):
        for key in w.keys():
            if i==0:
                w_avg[key] = ratio[i]*w[key]
            else:    
                w_avg[key]+=ratio[i]*w[key]
    return w_avg 

def add_net_state2(nets, ratio):
    """ implements the computation of model aggregatoin as per input ratio
        parties can be clients or clusters
        ratio is a list contains the coefficients assigned for each network
    """
    w_list = [net.state_dict() for net in nets]
    w_avg = w_list[0]
    for i, w in enumerate(w_list):
        for key in w.keys():
            if i==0:
                w_avg[key] = ratio[i]*w[key]
            else:    
                w_avg[key]+=ratio[i]*w[key]
    return w_avg


def add_net_state3(server_net, nets, ratio):
    """
        The idea is to update server_net with the weighted average of the nets.
        So nets contain incremental model changes with respect to the server_net.
    """
    server_net = deepcopy(server_net)
    diff_nets_with_server = [get_difference_between_network_weights(net, server_net) for net in nets]
    assert len(diff_nets_with_server) == len(ratio)
    w_avg = server_net.state_dict()
    for i, w in enumerate(diff_nets_with_server):
        for key in w.keys():
            w_avg[key] = w_avg[key] + w[key] * ratio[i]
    return w_avg

def get_difference_between_network_weights(net_1, net_2):
    """ compute the difference in the weight parameters of each layer of the two nets
    """
    weights_diff = {}
    
    # diff in parameters 
    # for (n, p), q in zip(net_1.named_parameters(), net_2.parameters()):
    #     weights_diff[n] = p.data.clone() - q.data.clone()

    # diff in state_dict
    net_1_state_dict = net_1.state_dict()
    net_2_state_dict = net_2.state_dict()
    for key in net_1_state_dict.keys():
        weights_diff[key] = net_1_state_dict[key] - net_2_state_dict[key]
    return weights_diff


def add_two_nets(source_net_1, source_net_2, target_net, alpha=0.5):
    """ this function add the weight parameters of two source networks, and assign the new weights to a target network
        the sum of any two weights is a weighted average using a ratio alpha
        note that the architecture of sources nets is the same with the target net 
    """
    weights_new = {}
    for (n, p), q in zip(source_net_1.named_parameters(), source_net_2.parameters()):
        weights_new[n] = (1-alpha) * p.data.clone() + alpha*q.data.clone()

    # overwrite the original networks weights with newly resampled ones    
    for n, p in target_net.named_parameters():
        p.data.copy_(weights_new[n])

def add_two_nets2(source_net_1, source_net_2, alpha=0.5):
    """ this function compute weighted average of the weight parameters of two source networks using a ratio alpha
        note that the architecture of sources nets must be the same
    """
    w_list = [source_net_1.state_dict(), source_net_2.state_dict()]
    ratio = [alpha, 1-alpha]
    w_avg = w_list[0]
    for i, w in enumerate(w_list):
        for key in w.keys():
            if i==0:
                w_avg[key] = ratio[i]*w[key]
            else:    
                w_avg[key]+=ratio[i]*w[key]
    return w_avg


def get_aggregated_model(nets, ratio):
    if(len(nets) == 0):
        return None
    w_avg = copy.deepcopy(nets[0])
    assert len(nets) == len(ratio), f"len(nets)={len(nets)}, len(ratio)={len(ratio)}"
    for i, w in enumerate(nets):
        for key in w.keys():
            if i==0:
                w_avg[key] = ratio[i] * w[key]
            else:    
                w_avg[key] = w_avg[key] + ratio[i] * w[key]
    return w_avg


def net_param_difference_dic(net_1, net_2):
    """ compute the difference in the weight parameters of each layer of the two nets
    """
    delta_norm_layer_dic = {}
    delta_norm_all_layer = 0
    for key in net_1.keys():
        delta_norm_layer_dic[key] = np.linalg.norm(net_1[key].cpu() - net_2[key].cpu())
        delta_norm_all_layer += delta_norm_layer_dic[key]**2
    delta_norm_all_layer = np.sqrt(delta_norm_all_layer)
    return delta_norm_all_layer


def net_param_difference(net_1, net_2):
    """ compute the difference in the weight parameters of each layer of the two nets
    """
    delta_norm_layer_dic = {}
    delta_norm_all_layer = 0
    delta_norm_fc = 0
    for (n, p), q in zip(net_1.named_parameters(), net_2.parameters()):
        delta_norm_this_layer = np.linalg.norm(p.data.cpu().clone() - q.data.cpu().clone())
        delta_norm_layer_dic[n] = delta_norm_this_layer
        delta_norm_all_layer += delta_norm_this_layer**2
        if n in ['classifier', 'Classifier', 'fc', 'f_c']:
            delta_norm_fc += delta_norm_this_layer**2
        
    delta_norm_embed = np.sqrt(delta_norm_all_layer - delta_norm_fc)
    delta_norm_all_layer = np.sqrt(delta_norm_all_layer)
    delta_norm_fc = np.sqrt(delta_norm_fc)
    return delta_norm_layer_dic, (delta_norm_all_layer, delta_norm_embed, delta_norm_fc)

def compare_model_param(model_0, model_1, input='tensor'):
    """ use to compare if the weight param. of two models are identical
    """
    flag=True
    if input=='tensor':
        for param_0, param_1 in zip(model_0, model_1):
            flag = flag * torch.equal(param_0.data, param_1.data)
            # flag = flag * (param_0.data == param_1.data)
    elif input=='dict':
        for name_0, name_1 in zip(model_0.keys(), model_1.keys()):
            flag = flag * torch.equal(model_0[name_0], model_1[name_1])
            if(flag == False):
                print(name_0, name_1)
                print(model_0[name_0] - model_1[name_1])
                print()
            # flag = flag * (model_0[name_0] == model_1[name_1])
    if flag:
        return True
    else:
        return False

@torch.no_grad()
def init_new_net(m):
    ''' using kaiming normal to randomly initialize networks to be trained
    '''
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)



# training and testing related utils
def evaluation(args, net, eval_loader):
    ''' Use this function to evalute the loss and accuracy of a model
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #net=nn.DataParallel(net).to(device)
    
    net=net.to(device)
    net.eval()
    #criterion = nn.CrossEntropyLoss(reduction='sum').to(args.device)
    criterion = nn.CrossEntropyLoss(reduction='sum').to(device)
    correct, loss = 0, 0.0
    n_test_samples = len(eval_loader.dataset)
    print('n_test_samples:', n_test_samples)

    for sample in eval_loader:
        print('evaluating now')
        img, labels, image_name = sample['image'], sample['label'], sample['image_name']
        img = Variable(img).to(device)
        labels = Variable(labels).long().to(device)

        #outputs = net(img).logits
        print('before net')
        outputs = net(img).logits
        print('after net')
         
            
         
        pred = outputs.argmax(dim=1, keepdim=True).squeeze()
        #pred = logits.argmax(-1).item()
        correct += pred.eq(labels.view_as(pred)).sum().item()
        loss += criterion(outputs, labels).item()
       

    print('After evaluation')    

    """ with torch.no_grad():
        for img in eval_loader:
            x = img['image'].to(device)
            y = img['label'].to(device)
            logits = net(x)
            pred = F.log_softmax(logits, dim=1)
            correct += (pred.argmax(dim=1) == y).sum().item()
            loss += criterion(pred,y).item() """
    
    """ with torch.no_grad():
        for x, y in eval_loader:
            #x, y = x.to(args.device), y.to(args.device)
            x, y = x.to(device), y.to(device)
            logits = net(x)
            pred = F.log_softmax(logits, dim=1)
            correct += (pred.argmax(dim=1) == y).sum().item()
            loss += criterion(pred,y).item() """
            
        
    if np.isnan(loss):
        print('loss is nan, print the model parameters')
        for name, param in net.state_dict().items():
            print(name, param)
        raise ValueError('loss is nan')

    acc, loss = correct / n_test_samples, loss / n_test_samples
    # if loss is nan, print the model parameters
    return acc, loss

def evaluation_statistical_parity(args, net, group_testloader):
    ''' Use this function to evalute the loss and accuracy of a model
    '''

    group_positive_pred = []
    for testloader in group_testloader:
        # evaluation of statistical parity
        net.eval()
        positive_pred = 0
        n_test_samples = len(testloader.dataset)
        
        with torch.no_grad():
            for x, y in testloader:
                x, y = x.to(args.device), y.to(args.device)
                logits = net(x)
                pred = F.log_softmax(logits, dim=1)
                positive_pred += (pred.argmax(dim=1) == 1).sum().item() # only consider positive prediction (class = 1)
        group_positive_pred.append(positive_pred / n_test_samples)

    group_positive_pred = np.array(group_positive_pred)

    return group_positive_pred.max() - group_positive_pred.min()



def evaluation_group_fairness(args, net, group_testloader):
    ''' Use this function to evalute the loss and accuracy of a model
    '''

    group_loss = []
    group_accuracy = []
    for testloader in group_testloader:
        acc, loss = evaluation(args, net, testloader)
        group_loss.append(loss)
        group_accuracy.append(acc)
    group_loss = np.array(group_loss)
    group_accuracy = np.array(group_accuracy)

    return group_accuracy.max() - group_accuracy.min(), group_loss.max() - group_loss.min()
    

def get_metrics(args, net, eval_loader):
    ''' used for normal FL training evaluation
    '''

    net.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum').to(args.device)
    
    with torch.no_grad():
        logits_all, targets_all = torch.tensor([], device='cpu'), torch.tensor([], dtype=torch.int, device='cpu')
        for x, y in eval_loader:
            x, y = x.to(args.device), y.to(args.device)
            logits = net(x)
            logits_all = torch.cat((logits_all, logits.detach().cpu()),dim=0)
            targets_all = torch.cat((targets_all, y.cpu()), dim=0)
    
    pred = F.log_softmax(logits_all, dim=1)
    loss = criterion(pred, targets_all)/len(eval_loader.dataset) # validation loss
    
    output = pred.argmax(dim=1) # predicated/output label
    prob = F.softmax(logits_all, dim=1) # probabilities

    acc = accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
    bal_acc = balanced_accuracy_score(y_pred=output.numpy(), y_true=targets_all.numpy())
    auc = roc_auc_score(targets_all.numpy(), prob.numpy(), multi_class='ovr')

    return loss, acc, auc, bal_acc


def compute_accuracy(model, dataloader, get_confusion_matrix=False, moon_model=False, device="cpu"):
    r""" the function to evaluate model accuracy used by original open-source code 
    """

    was_training = False
    if model.training:
        model.eval()
        was_training = True

    true_labels_list, pred_labels_list = np.array([]), np.array([])

    if type(dataloader) == type([1]):
        pass
    else:
        dataloader = [dataloader]

    correct, total = 0, 0
    with torch.no_grad():
        for tmp in dataloader:
            for batch_idx, (x, target) in enumerate(tmp):
                x, target = x.to(device), target.to(device,dtype=torch.int64)
                if moon_model:
                    _, _, out = model(x)
                else:
                    out = model(x)
                _, pred_label = torch.max(out.data, 1)

                total += x.data.size()[0]
                correct += (pred_label == target.data).sum().item()

                if device == "cpu":
                    pred_labels_list = np.append(pred_labels_list, pred_label.numpy())
                    true_labels_list = np.append(true_labels_list, target.data.numpy())
                else:
                    pred_labels_list = np.append(pred_labels_list, pred_label.cpu().numpy())
                    true_labels_list = np.append(true_labels_list, target.data.cpu().numpy())

    if get_confusion_matrix:
        conf_matrix = confusion_matrix(true_labels_list, pred_labels_list)

    if was_training:
        model.train()

    if get_confusion_matrix:
        return correct/float(total), conf_matrix

    return correct/float(total)

def plot_series_mul(series, y_min=0.5, y_max=1.0, title='Test acc', step=1, save=False, save_path=None):
    """ input series must be a numpy array
    """
    
    colors = ['crimson', 'gold', 'deepskyblue', 'limegreen', 'deeppink', 'darkorange', 'sienna', 'slategrey']
    linestyles = ['-', '--', ':', '-.']
    
    run_colors, run_linestyles = [0]*len(series), [0]*len(series)
    plt.figure(figsize=(6,4),dpi=200)
    for i, run in enumerate(series):
        run_colors[i]=colors[i]
        if i < len(colors):
            run_linestyles[i]=linestyles[0]
        elif i < 2*len(colors):
            run_linestyles[i]=linestyles[1]
        elif i < 3*len(colors):
            run_linestyles[i]=linestyles[2]
        elif i < 4*len(colors):
            run_linestyles[i]=linestyles[3]
        # records[run] = torch.tensor([series[i] for i in range(len(series))]).mean(0).tolist() # for later use to average over multiple independent runs

        lc = series[i]
        steps = torch.tensor(range(1,len(lc)+1)).tolist()
        data_x = [x for j, x in enumerate(steps) if (j+1) % step == 0]
        data_y = [y for j, y in enumerate(lc) if (j+1) % step==0]
        plt.plot(data_x,data_y,label='exp_{:d}'.format(i), linestyle=run_linestyles[i], color=run_colors[i], linewidth=1)
        plt.legend(loc='lower right')
        
    plt.grid()
    # plt.tight_layout()
    plt.xlim(1,len(lc)+1)
    plt.xticks([int(i+1) for i in range(len(lc))])
    plt.ylim(y_min,y_max)
    # plt.yticks([.76, .78, .80, .82, .84, .86, .88, .90, .92, .94])
    plt.xlabel('Comm. rounds')
    plt.ylabel(title, size='large')
    # plt.title('Adam, lr=1e-4, local epoch E=2, batch size=32')

    # save figures
    # save_fig_path = './saved_fig/' + time.strftime('%y-%m-%d-%H-%M-%S.png')
    if save:
        plt.savefig(save_path)

def plot_series(
    series, 
    y_min=0, 
    y_max=1.0, 
    series_name='series_name', 
    title='Test acc', 
    step=1, 
    save_path=None
):
    """ input series must be a 1-d numpy array, e.g., the mean accuracy time series
    """
    series_len = series.shape[-1]
    colors = ['crimson', 'gold', 'deepskyblue', 'limegreen', 'deeppink', 'darkorange', 'sienna', 'slategrey']
    linestyles = ['-', '--', ':', '-.']
    
    run_colors, run_linestyles = colors[0], linestyles[0]
    plt.figure(figsize=(6,4),dpi=200)
    # steps = torch.tensor(range(1,series_len+1)).tolist()
    # data_x = [x for j, x in enumerate(steps) if (j+1) % step == 0]
    data_y = [y for j, y in enumerate(series) if (j+1) % step==0]
    plt.plot(data_y,label=series_name, linestyle=run_linestyles, color=run_colors, linewidth=1)
    plt.legend(loc='lower right')
        
    plt.grid()
    # plt.tight_layout()
    plt.xlim(0,series_len+1)
    # plt.xticks([int(i+1) for i in range(series_len)])
    plt.ylim(y_min,y_max)
    plt.xlabel('Comm. rounds')
    plt.ylabel(title, size='large')

    if save_path is not None:
        plt.savefig(save_path)

def make_learning_curve(x):
    """ Use this function to get monotonically incerasing learning curve
        from an input list x whose elements are accuracy points
    """
    running_max=0
    y=[0]*len(x)
    for idx, value in enumerate(x):
        running_max = max(x[:idx+1])
        y[idx] = running_max if x[idx] < running_max else x[idx]
    return y

# additional for fed-dc
def get_irm_loss(target_mat, src_mat):
    r"""
    Use this method to get the regularization loss term using inter-client relationship matching (IRM) technique reported in https://arxiv.org/abs/2106.08600
    """        
    return (F.kl_div(src_mat.log(), target_mat, None, None, 'batchmean') + F.kl_div(target_mat.log(), target_mat, None, None, 'batchmean'))/2.0

class CKA_Torch(object):
    def __init__(self, device):
        self.device = device
    
    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)  

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.T)
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = math.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def linear_HSIC(self, X, Y):
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        return hsic / (var1 * var2)

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))
        return hsic / (var1 * var2)

def rand_projections(embedding_dim, num_samples=50):
    """This function generates `num_samples` random samples from the latent space's unit sphere.

        Args:
            embedding_dim (int): embedding dimensionality
            num_samples (int): number of random projection samples

        Return:
            torch.Tensor: tensor of size (num_samples, embedding_dim)
    """
    projections = [w / np.sqrt((w**2).sum())  # L2 normalization
                   for w in np.random.normal(size=(num_samples, embedding_dim))]
    projections = np.asarray(projections)
    return torch.from_numpy(projections).type(torch.FloatTensor)


def sliced_wasserstein_distance(
        encoded_samples,
        distribution_samples,
        num_projections=50,
        p=2,
        device='cpu'
    ):
    r""" Sliced Wasserstein Distance between encoded samples and drawn distribution samples.

    Adapted from https://github.com/koshian2/swd-pytorch

        Args:
            encoded_samples (toch.Tensor): tensor of encoded training samples
            distribution_samples (torch.Tensor): tensor of drawn distribution training samples
            num_projections (int): number of projections to approximate sliced wasserstein distance
            p (int): power of distance metric
            device (torch.device): torch device (default 'cpu')

        Return:
            torch.Tensor: tensor of wasserstrain distances of size (num_projections, 1)
    """
    # derive latent space dimension size from random samples drawn from latent prior distribution
    embedding_dim = distribution_samples.size(1)

    # generate random projections in latent space
    projections = rand_projections(embedding_dim, num_projections).to(device)

    # calculate projections through the encoded samples
    encoded_projections = encoded_samples.matmul(projections.transpose(0, 1))
   
    # calculate projections through the prior distribution random samples
    distribution_projections = (distribution_samples.matmul(projections.transpose(0, 1)))
    
    # calculate the sliced wasserstein distance by
    # sorting the samples per random projection and
    # calculating the difference between the
    # encoded samples and drawn random samples
    # per random projection
    wasserstein_distance = (torch.sort(encoded_projections.transpose(0, 1), dim=1)[0] -
                            torch.sort(distribution_projections.transpose(0, 1), dim=1)[0])
    # distance between latent space prior and encoded distributions
    # power of 2 by default for Wasserstein-2
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    # approximate mean wasserstein_distance for each projection
    return wasserstein_distance.mean()

# # KL divergence for multivariate Gaussians
# def kl_div_dist(mu1, Sigma1, mu2, Sigma2):
#     k = mu1.shape[0]
#     diff = mu2 - mu1
#     tr_term = torch.trace(torch.matmul(torch.inverse(Sigma2), Sigma1))
#     quad_term = torch.matmul(torch.matmul(diff.T, torch.inverse(Sigma2)), diff)
#     log_det_term = torch.log(torch.det(Sigma2) / torch.det(Sigma1))
#     return 0.5 * (tr_term + quad_term - k + log_det_term)

# # Wasserstein distance for multivariate Gaussians
# def wass_dist(mu1, Sigma1, mu2, Sigma2):
#     mean_diff = torch.norm(mu1 - mu2)**2
#     cov_diff = torch.trace(Sigma1 + Sigma2 - 2 * torch.from_numpy(sqrtm(sqrtm(Sigma1.detach().cpu().numpy()) @ Sigma2.detach().cpu().numpy() @ sqrtm(Sigma1.detach().cpu().numpy()))))
#     return mean_diff + cov_diff

# # Calculate covariance using PyTorch
# def get_cov(features):
#     feature_mean = torch.mean(features, dim=0)
#     feature_diff = features - feature_mean.unsqueeze(0)
#     covariance = torch.matmul(feature_diff.T, feature_diff) / (features.shape[0] - 1)
#     return covariance


# all the rest is for data condensation purpose
class TensorDataset(Dataset):
    """ 05 Jan 2023: added transform in __getitem__() so that server can apply server-side transform before the training of federated model
    """
    def __init__(self, images, labels, transform=None): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()
        self.transform = transform

    def __getitem__(self, index):
        images, labels = self.images[index], self.labels[index]
        if self.transform is not None:
            images = self.transform(images)
        return images, labels

    def __len__(self):
        return self.images.shape[0]

class ServerSynDataset(Dataset):
    """ Use this class to create server-side dataset for the collected syn data
        Along with the images and lables, added src_id to indicate the id of source client
        05 Jan 2023: added transform in __getitem__() so that server can apply server-side transform before the training of federated model
    """
    def __init__(self, images, labels, src, transform=None): # images: n x c x h x w tensor
        self.images = images.detach().float()
        self.labels = labels.detach()
        self.src = src
        self.transform = transform

    def __getitem__(self, index):
        images, labels = self.images[index], self.labels[index]
        if self.transform is not None:
            images = self.transform(images)
        src_id = self.src[index]
        return images, labels, src_id

    def __len__(self):
        return self.images.shape[0]

def get_default_convnet_setting():
    net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
    return net_width, net_depth, net_act, net_norm, net_pooling



def get_network(model, channel, num_classes, im_size=(32, 32)):
    torch.random.manual_seed(int(time.time() * 1000) % 100000)
    net_width, net_depth, net_act, net_norm, net_pooling = get_default_convnet_setting()

    if model == 'MLP':
        net = MLP(channel=channel, num_classes=num_classes)
    elif model == 'MLP_tabular':
        net = MLP_tabular(input_size=im_size, num_classes=num_classes)
    elif model == 'MLP_linear':
        net = MLP_linear(input_size=im_size, num_classes=num_classes) 
    elif model == 'ConvNet':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'LeNet':
        net = LeNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNet':
        net = AlexNet(channel=channel, num_classes=num_classes)
    elif model == 'AlexNetBN':
        net = AlexNetBN(channel=channel, num_classes=num_classes)
    elif model == 'VGG11':
        net = VGG11( channel=channel, num_classes=num_classes)
    elif model == 'VGG11BN':
        net = VGG11BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18':
        net = ResNet18(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN_AP':
        net = ResNet18BN_AP(channel=channel, num_classes=num_classes)
    elif model == 'ResNet18BN':
        net = ResNet18BN(channel=channel, num_classes=num_classes)
    elif model == 'ResNet50':
        net = ResNet50(channel=channel, num_classes=num_classes)

    elif model == 'ConvNetD1':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=1, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD2':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=2, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD3':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=3, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetD4':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=4, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetW32':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=32, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW64':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=64, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW128':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=128, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetW256':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=256, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetAS':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='sigmoid', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAR':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='relu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetAL':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='leakyrelu', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwish':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm=net_norm, net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetASwishBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act='swish', net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='none', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetBN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='batchnorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetLN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='layernorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetIN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='instancenorm', net_pooling=net_pooling, im_size=im_size)
    elif model == 'ConvNetGN':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm='groupnorm', net_pooling=net_pooling, im_size=im_size)

    elif model == 'ConvNetNP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='none', im_size=im_size)
    elif model == 'ConvNetMP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='maxpooling', im_size=im_size)
    elif model == 'ConvNetAP':
        net = ConvNet(channel=channel, num_classes=num_classes, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling='avgpooling', im_size=im_size)

    else:
        net = None
        exit('unknown model: %s'%model)

    gpu_num = torch.cuda.device_count()
    if gpu_num>0:
        device = 'cuda'
        if gpu_num>1:
            net = nn.DataParallel(net)
    else:
        device = 'cpu'
    net = net.to(device)

    return net



def distance_wb(gwr, gws):
    shape = gwr.shape
    if len(shape) == 4: # conv, out*in*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2] * shape[3])
        gws = gws.reshape(shape[0], shape[1] * shape[2] * shape[3])
    elif len(shape) == 3:  # layernorm, C*h*w
        gwr = gwr.reshape(shape[0], shape[1] * shape[2])
        gws = gws.reshape(shape[0], shape[1] * shape[2])
    elif len(shape) == 2: # linear, out*in
        tmp = 'do nothing'
    elif len(shape) == 1: # batchnorm/instancenorm, C; groupnorm x, bias
        gwr = gwr.reshape(1, shape[0])
        gws = gws.reshape(1, shape[0])
        return torch.tensor(0, dtype=torch.float, device=gwr.device)

    dis_weight = torch.sum(1 - torch.sum(gwr * gws, dim=-1) / (torch.norm(gwr, dim=-1) * torch.norm(gws, dim=-1) + 0.000001))
    dis = dis_weight
    return dis



def match_loss(gw_syn, gw_real, args):
    dis = torch.tensor(0.0).to(args.device)

    if args.dis_metric == 'ours':
        for ig in range(len(gw_real)):
            gwr = gw_real[ig]
            gws = gw_syn[ig]
            dis += distance_wb(gwr, gws)

    elif args.dis_metric == 'mse':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = torch.sum((gw_syn_vec - gw_real_vec)**2)

    elif args.dis_metric == 'cos':
        gw_real_vec = []
        gw_syn_vec = []
        for ig in range(len(gw_real)):
            gw_real_vec.append(gw_real[ig].reshape((-1)))
            gw_syn_vec.append(gw_syn[ig].reshape((-1)))
        gw_real_vec = torch.cat(gw_real_vec, dim=0)
        gw_syn_vec = torch.cat(gw_syn_vec, dim=0)
        dis = 1 - torch.sum(gw_real_vec * gw_syn_vec, dim=-1) / (torch.norm(gw_real_vec, dim=-1) * torch.norm(gw_syn_vec, dim=-1) + 0.000001)

    else:
        exit('unknown distance function: %s'%args.dis_metric)

    return dis


# def get_loops(ipc):
#     # Get the two hyper-parameters of outer-loop and inner-loop.
#     # The following values are empirically good.
#     if ipc == 1:
#         outer_loop, inner_loop = 1, 1
#     elif ipc == 10:
#         outer_loop, inner_loop = 10, 50
#     elif ipc == 20:
#         outer_loop, inner_loop = 20, 25
#     elif ipc == 30:
#         outer_loop, inner_loop = 30, 20
#     elif ipc == 40:
#         outer_loop, inner_loop = 40, 15
#     elif ipc == 50:
#         outer_loop, inner_loop = 50, 10
#     else:
#         outer_loop, inner_loop = 0, 0
#         exit('loop hyper-parameters are not defined for %d ipc'%ipc)
#     return outer_loop, inner_loop



def epoch(mode, dataloader, net, optimizer, criterion, args, aug):
    ''' 
    Notes on 15-Jun-2023:

    This function is moved to server.py as a method of ClientDC.
    '''
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)

    if mode == 'train':
        net.train()
    else:
        net.eval()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=ParamDiffAug())
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].long().to(args.device)
        n_b = lab.shape[0]

        output = net(img)
        loss = criterion(output, lab)
        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))

        loss_avg += loss.item()*n_b
        acc_avg += acc
        num_exp += n_b

        if mode == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def epoch_data_weight(dataloader, net, optimizer, criterion, args, aug, data_weights=None):
    ''' Introduction:  
        1) this method just duplicate the original method "epoch", but only used for train purpose and also to include some custmoized functions.
        2) argument src_id_lut is should be provided with server.syn_data_src which is a tensor where entry indicates the id of the clients from which the data is collected, 
        this argument is used for weighting the data with the next argument data_weights
        3) argument data_weights is the weights associated to each training sample
    '''
    loss_avg, acc_avg, num_exp = 0, 0, 0
    net = net.to(args.device)
    criterion = criterion.to(args.device)
    net.train()

    for i_batch, datum in enumerate(dataloader):
        img = datum[0].float().to(args.device)
        if aug:
            if args.dsa:
                img = DiffAugment(img, args.dsa_strategy, param=ParamDiffAug())
            else:
                img = augment(img, args.dc_aug_param, device=args.device)
        lab = datum[1].to(args.device)
        n_b = lab.shape[0]
        src_id = datum[2]

        output = net(img)
        loss_batch = criterion(output, lab) # this loss is a vector, with reduction = none
        if data_weights is not None:
            weights = torch.tensor([data_weights[lab[i]][src_id[i]] for i in range(n_b)]).to(args.device)
            loss = torch.sum(loss_batch*weights)/n_b
            # loss = torch.sum(loss_batch*weights)/torch.sum(weights) # however, torch.sum(weights) = 1.0 if you apply normalization to cust weights into [0,1]    
        else:
            loss = torch.mean(loss_batch) # this line is equivalent to using nn.CrossEntropy(reduction='mean')
        loss_avg += loss.item()*n_b

        acc = np.sum(np.equal(np.argmax(output.cpu().data.numpy(), axis=-1), lab.cpu().data.numpy()))
        acc_avg += acc
        num_exp += n_b

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_avg /= num_exp
    acc_avg /= num_exp

    return loss_avg, acc_avg

def augment(images, dc_aug_param, device):
    # This can be sped up in the future.

    if dc_aug_param != None and dc_aug_param['strategy'] != 'none':
        scale = dc_aug_param['scale']
        crop = dc_aug_param['crop']
        rotate = dc_aug_param['rotate']
        noise = dc_aug_param['noise']
        strategy = dc_aug_param['strategy']

        shape = images.shape
        mean = []
        for c in range(shape[1]):
            mean.append(float(torch.mean(images[:,c])))

        def cropfun(i):
            im_ = torch.zeros(shape[1],shape[2]+crop*2,shape[3]+crop*2, dtype=torch.float, device=device)
            for c in range(shape[1]):
                im_[c] = mean[c]
            im_[:, crop:crop+shape[2], crop:crop+shape[3]] = images[i]
            r, c = np.random.permutation(crop*2)[0], np.random.permutation(crop*2)[0]
            images[i] = im_[:, r:r+shape[2], c:c+shape[3]]

        def scalefun(i):
            h = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            w = int((np.random.uniform(1 - scale, 1 + scale)) * shape[2])
            tmp = F.interpolate(images[i:i + 1], [h, w], )[0]
            mhw = max(h, w, shape[2], shape[3])
            im_ = torch.zeros(shape[1], mhw, mhw, dtype=torch.float, device=device)
            r = int((mhw - h) / 2)
            c = int((mhw - w) / 2)
            im_[:, r:r + h, c:c + w] = tmp
            r = int((mhw - shape[2]) / 2)
            c = int((mhw - shape[3]) / 2)
            images[i] = im_[:, r:r + shape[2], c:c + shape[3]]

        def rotatefun(i):
            im_ = scipyrotate(images[i].cpu().data.numpy(), angle=np.random.randint(-rotate, rotate), axes=(-2, -1), cval=np.mean(mean))
            r = int((im_.shape[-2] - shape[-2]) / 2)
            c = int((im_.shape[-1] - shape[-1]) / 2)
            images[i] = torch.tensor(im_[:, r:r + shape[-2], c:c + shape[-1]], dtype=torch.float, device=device)

        def noisefun(i):
            images[i] = images[i] + noise * torch.randn(shape[1:], dtype=torch.float, device=device)


        augs = strategy.split('_')

        for i in range(shape[0]):
            choice = np.random.permutation(augs)[0] # randomly implement one augmentation
            if choice == 'crop':
                cropfun(i)
            elif choice == 'scale':
                scalefun(i)
            elif choice == 'rotate':
                rotatefun(i)
            elif choice == 'noise':
                noisefun(i)

    return images



def get_daparam(dataset, model, model_eval, ipc):
    # We find that augmentation doesn't always benefit the performance.
    # So we do augmentation for some of the settings.

    dc_aug_param = dict()
    dc_aug_param['crop'] = 4
    dc_aug_param['scale'] = 0.2
    dc_aug_param['rotate'] = 45
    dc_aug_param['noise'] = 0.001
    dc_aug_param['strategy'] = 'none'

    if dataset == 'MNIST':
        dc_aug_param['strategy'] = 'crop_scale_rotate'

    if model_eval in ['ConvNetBN']: # Data augmentation makes model training with Batch Norm layer easier.
        dc_aug_param['strategy'] = 'crop_noise'

    return dc_aug_param


def get_eval_pool(eval_mode, model, model_eval):
    if eval_mode == 'M': # multiple architectures
        model_eval_pool = ['MLP', 'ConvNet', 'LeNet', 'AlexNet', 'VGG11', 'ResNet18']
    elif eval_mode == 'B':  # multiple architectures with BatchNorm for DM experiments
        model_eval_pool = ['ConvNetBN', 'ConvNetASwishBN', 'AlexNetBN', 'VGG11BN', 'ResNet18BN']
    elif eval_mode == 'W': # ablation study on network width
        model_eval_pool = ['ConvNetW32', 'ConvNetW64', 'ConvNetW128', 'ConvNetW256']
    elif eval_mode == 'D': # ablation study on network depth
        model_eval_pool = ['ConvNetD1', 'ConvNetD2', 'ConvNetD3', 'ConvNetD4']
    elif eval_mode == 'A': # ablation study on network activation function
        model_eval_pool = ['ConvNetAS', 'ConvNetAR', 'ConvNetAL', 'ConvNetASwish']
    elif eval_mode == 'P': # ablation study on network pooling layer
        model_eval_pool = ['ConvNetNP', 'ConvNetMP', 'ConvNetAP']
    elif eval_mode == 'N': # ablation study on network normalization layer
        model_eval_pool = ['ConvNetNN', 'ConvNetBN', 'ConvNetLN', 'ConvNetIN', 'ConvNetGN']
    elif eval_mode == 'S': # itself
        if 'BN' in model:
            print('Attention: Here I will replace BN with IN in evaluation, as the synthetic set is too small to measure BN hyper-parameters.')
        model_eval_pool = [model[:model.index('BN')]] if 'BN' in model else [model]
    elif eval_mode == 'SS':  # itself
        model_eval_pool = [model]
    else:
        model_eval_pool = [model_eval]
    return model_eval_pool


class ParamDiffAug():
    def __init__(self):
        self.aug_mode = 'S' #'multiple or single'
        self.prob_flip = 0.5
        self.ratio_scale = 1.2
        self.ratio_rotate = 15.0
        self.ratio_crop_pad = 0.125
        self.ratio_cutout = 0.5 # the size would be 0.5x0.5
        self.brightness = 1.0
        self.saturation = 2.0
        self.contrast = 0.5


def set_seed_DiffAug(param):
    if param.latestseed == -1:
        return
    else:
        torch.random.manual_seed(param.latestseed)
        param.latestseed += 1


def DiffAugment(x, strategy='', seed = -1, param = None):
    if strategy == 'None' or strategy == 'none' or strategy == '':
        return x

    if seed == -1:
        param.Siamese = False
    else:
        param.Siamese = True

    param.latestseed = seed

    if strategy:
        if param.aug_mode == 'M': # original
            for p in strategy.split('_'):
                for f in AUGMENT_FNS[p]:
                    x = f(x, param)
        elif param.aug_mode == 'S':
            pbties = strategy.split('_')
            set_seed_DiffAug(param)
            p = pbties[torch.randint(0, len(pbties), size=(1,)).item()]
            for f in AUGMENT_FNS[p]:
                x = f(x, param)
        else:
            exit('unknown augmentation mode: %s'%param.aug_mode)
        x = x.contiguous()
    return x


# We implement the following differentiable augmentation strategies based on the code provided in https://github.com/mit-han-lab/data-efficient-gans.
def rand_scale(x, param):
    # x>1, max scale
    # sx, sy: (0, +oo), 1: orignial size, 0.5: enlarge 2 times
    ratio = param.ratio_scale
    set_seed_DiffAug(param)
    sx = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    set_seed_DiffAug(param)
    sy = torch.rand(x.shape[0]) * (ratio - 1.0/ratio) + 1.0/ratio
    theta = [[[sx[i], 0,  0],
            [0,  sy[i], 0],] for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_rotate(x, param): # [-180, 180], 90: anticlockwise 90 degree
    ratio = param.ratio_rotate
    set_seed_DiffAug(param)
    theta = (torch.rand(x.shape[0]) - 0.5) * 2 * ratio / 180 * float(np.pi)
    theta = [[[torch.cos(theta[i]), torch.sin(-theta[i]), 0],
        [torch.sin(theta[i]), torch.cos(theta[i]),  0],]  for i in range(x.shape[0])]
    theta = torch.tensor(theta, dtype=torch.float)
    if param.Siamese: # Siamese augmentation:
        theta[:] = theta[0]
    grid = F.affine_grid(theta, x.shape).to(x.device)
    x = F.grid_sample(x, grid)
    return x


def rand_flip(x, param):
    prob = param.prob_flip
    set_seed_DiffAug(param)
    randf = torch.rand(x.size(0), 1, 1, 1, device=x.device)
    if param.Siamese: # Siamese augmentation:
        randf[:] = randf[0]
    return torch.where(randf < prob, x.flip(3), x)


def rand_brightness(x, param):
    ratio = param.brightness
    set_seed_DiffAug(param)
    randb = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randb[:] = randb[0]
    x = x + (randb - 0.5)*ratio
    return x


def rand_saturation(x, param):
    ratio = param.saturation
    x_mean = x.mean(dim=1, keepdim=True)
    set_seed_DiffAug(param)
    rands = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        rands[:] = rands[0]
    x = (x - x_mean) * (rands * ratio) + x_mean
    return x


def rand_contrast(x, param):
    ratio = param.contrast
    x_mean = x.mean(dim=[1, 2, 3], keepdim=True)
    set_seed_DiffAug(param)
    randc = torch.rand(x.size(0), 1, 1, 1, dtype=x.dtype, device=x.device)
    if param.Siamese:  # Siamese augmentation:
        randc[:] = randc[0]
    x = (x - x_mean) * (randc + ratio) + x_mean
    return x


def rand_crop(x, param):
    # The image is padded on its surrounding and then cropped.
    ratio = param.ratio_crop_pad
    shift_x, shift_y = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    translation_x = torch.randint(-shift_x, shift_x + 1, size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        translation_x[:] = translation_x[0]
        translation_y[:] = translation_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(x.size(2), dtype=torch.long, device=x.device),
        torch.arange(x.size(3), dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + translation_x + 1, 0, x.size(2) + 1)
    grid_y = torch.clamp(grid_y + translation_y + 1, 0, x.size(3) + 1)
    x_pad = F.pad(x, [1, 1, 1, 1, 0, 0, 0, 0])
    x = x_pad.permute(0, 2, 3, 1).contiguous()[grid_batch, grid_x, grid_y].permute(0, 3, 1, 2)
    return x


def rand_cutout(x, param):
    ratio = param.ratio_cutout
    cutout_size = int(x.size(2) * ratio + 0.5), int(x.size(3) * ratio + 0.5)
    set_seed_DiffAug(param)
    offset_x = torch.randint(0, x.size(2) + (1 - cutout_size[0] % 2), size=[x.size(0), 1, 1], device=x.device)
    set_seed_DiffAug(param)
    offset_y = torch.randint(0, x.size(3) + (1 - cutout_size[1] % 2), size=[x.size(0), 1, 1], device=x.device)
    if param.Siamese:  # Siamese augmentation:
        offset_x[:] = offset_x[0]
        offset_y[:] = offset_y[0]
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(x.size(0), dtype=torch.long, device=x.device),
        torch.arange(cutout_size[0], dtype=torch.long, device=x.device),
        torch.arange(cutout_size[1], dtype=torch.long, device=x.device),
    )
    grid_x = torch.clamp(grid_x + offset_x - cutout_size[0] // 2, min=0, max=x.size(2) - 1)
    grid_y = torch.clamp(grid_y + offset_y - cutout_size[1] // 2, min=0, max=x.size(3) - 1)
    mask = torch.ones(x.size(0), x.size(2), x.size(3), dtype=x.dtype, device=x.device)
    mask[grid_batch, grid_x, grid_y] = 0
    x = x * mask.unsqueeze(1)
    return x


AUGMENT_FNS = {
    'color': [rand_brightness, rand_saturation, rand_contrast],
    'crop': [rand_crop],
    'cutout': [rand_cutout],
    'flip': [rand_flip],
    'scale': [rand_scale],
    'rotate': [rand_rotate],
}

# used for benchmark.py
def get_dataloader(trainset, testset, train_bs, test_bs, dataidxs=None, transform=None):
    """ used for benchmark algorithms
    """
    if dataidxs is not None:
        # split for clients local train and test dataset
        train_ds = CustomSubset(dataset=trainset, indices=dataidxs, subset_transform=transform)
        test_ds = CustomSubset(dataset=testset, indices=dataidxs, subset_transform=transform)
    else:
        # using pooled centralized dataset
        train_ds = trainset
        test_ds = testset

    train_dl = DataLoader(dataset=train_ds, batch_size=train_bs, shuffle=True, drop_last=False)
    test_dl = DataLoader(dataset=test_ds, batch_size=test_bs, shuffle=False, drop_last=False)

    return train_dl, test_dl, train_ds, test_ds
