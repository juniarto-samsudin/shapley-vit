from . datasets.dataloader_cell import XrayDataLoader as CellDataLoader
from . federated_learning.utils import evaluation, get_dataset, get_difference_between_network_weights
from . models.xray_inception_network import inception_network
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torch.autograd import Variable
from torch import nn
import torch.optim as optim
import torch as th
import os
import time
import errno
from . fed_client_contribution.game2 import Game
from . fed_client_contribution.utils_shapley import call_shapley_computation_method
from . federated_learning.client2 import ClientBase
from . federated_learning.server2 import ServerBase
from transformers import ViTFeatureExtractor, ViTModel, ViTForImageClassification,ViTImageProcessor, AutoModel
from dotenv import load_dotenv
import re
import pandas as pd
import logging
import redis
import numpy as np
import json
import sys 

load_dotenv()
session_id = os.getenv("SESSION_ID")
party_id0 = os.getenv("PARTY_ID0")
party_id1 = os.getenv("PARTY_ID1")
party_id2 = os.getenv("PARTY_ID2")
myUserMap = {0: party_id0, 
             1: party_id1, 
             2: party_id2
            }
log_name = 'container-{}.log'.format(session_id)
logging.basicConfig(filename=("./logs/container-logs/{}".format(log_name)), 
                    level=logging.DEBUG, 
                    format='%(asctime)s %(levelname)s %(message)s', 
                    datefmt='%m/%d/%Y %I:%M:%S %p')

#Get Redis Host from environment variable in docker-compose
#If not found, use localhost for development
redis_host = os.getenv('REDIS_HOST', 'localhost')
redis_port = os.getenv('REDIS_PORT', 6379)
logging.info('Redis Host: {}'.format(redis_host))
logging.info('Session ID: {}'.format(session_id))
r = redis.Redis(host=redis_host, port=redis_port, db=0, decode_responses=True)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_local_model_path1 = os.getenv("LOCAL_MODEL_PATH1")
my_local_model_path2 = os.getenv("LOCAL_MODEL_PATH2")
my_local_model_path3 = os.getenv("LOCAL_MODEL_PATH3")
my_global_model_path = os.getenv("GLOBAL_MODEL_PATH")
my_validation_dataset = os.getenv("VALIDATION_DATASET")

logging.info('my_validation_dataset: {}'.format(my_validation_dataset))

import re

def natural_keys(text):
    """
    A helper function that returns a list of either integers or text components from a string.
    For example, 'ViT_epoch_10.pth.tar' -> ['ViT_epoch_', 10, '.pth.tar']
    """
    return [int(c) if c.isdigit() else c for c in re.split('([0-9]+)', text)]

def getOCTData():
    root_dir = '/mnt/data/home/astar/FL_Platform_crypten/OCT/CellData/OCT1/train'
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=root_dir, transform=transform)
    
    # Get the indices of the dataset
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, stratify=dataset.targets)

    # Create subsets for training and testing
    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    data_set = {}
    data_set['train_data'] = train_dataset
    return data_set

def getOCTData2():
    dataset = CellDataLoader(root_dir = my_validation_dataset,
                              mode = 'train',
                              patch_size=256,
                              sub_dir='')
    return dataset

def train(dataset):
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = inception_network().to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    lr = 0.001

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), \
                                   lr=lr, betas=(0.9, 0.999),
                                   weight_decay=0.0005)

    for sample in dataset:
        img, labels = sample['image'], sample['label']
        img = Variable(img).to(device)
        labels = Variable(labels).to(device)
        outputs = model(img).to(device)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        pred = outputs.argmax(dim=1, keepdim=True).squeeze()
        correct = pred.eq(labels.view_as(pred)).sum().item()

    return None

def getInitialShapleyValue(dataset, init_global_model, client_model_1, client_model_2, client_model_3):
    args={}
    valid_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
    fed_valid_acc, fed_valid_loss = evaluation(args, init_global_model, valid_loader)

    previous_utility = []
    utility_map = {
        0: 'accuracy',
        1: 'loss'
    }
    utility_dim = len(utility_map)

    previous_utility.append(fed_valid_acc)
    previous_utility.append(fed_valid_loss)
    logging.info('Previous utility: {}'.format(previous_utility)) #[0.21171171171171171, 1.5319562344937712]

    shapley_value_all_rounds = [[] for _ in range(utility_dim)]  #[[], []]
    shapley_value_sum = [{} for _ in range(utility_dim)]         #[{}, {}]

    num_clients=3
  
    # Initial Shapley value
    for i in range(utility_dim):
        shapley_value_all_rounds[i].append({client_id: previous_utility[i] / num_clients for client_id in range(num_clients)})
        shapley_value_sum[i] = shapley_value_all_rounds[i][0]
    
    logging.info('shapley_value_all_rounds: {}'.format(shapley_value_all_rounds))
    """ 
    [
        [{0: 0.059159159159159154, 1: 0.059159159159159154, 2: 0.059159159159159154}], ---> accuracy 
        [{0: 0.5220634746837902, 1: 0.5220634746837902, 2: 0.5220634746837902}]        ---> loss 
    ]
    """
    logging.info('shapley_value_sum: {}'.format(shapley_value_sum))
    """
    [
        {0: 0.059159159159159154, 1: 0.059159159159159154, 2: 0.059159159159159154},    ---> accuracy
        {0: 0.5220634746837902, 1: 0.5220634746837902, 2: 0.5220634746837902}           ---> loss
    ]
    """

    #client local training
    #monitoring client directories
    client_processed_model1 = []
    client_processed_model2 = []
    client_processed_model3 = []
    global_processed_model = [] 
    #print('client_model_list1: ', client_model_list1)
    #print('global_model_list: ', global_model_list)
    acc_session_all = []
    loss_session_all = []
    shapley_session_all = []

    # create clients
    clients_all = [ClientBase(id, args, init_global_model, dataset)
                            for id in range(num_clients)] 

    # create the server
    server = ServerBase(args, init_global_model, clients_all,None, valid_loader, None)
    break_all = False
    continueCount = 0
    globalEpochCounter = 0
    #while len(client_processed_model1) <= 1 and len(client_processed_model2) <= 1 and len(client_processed_model3) <= 1: #process 1 model only for each client
    while True:
        client_model_list1 = [f for f in os.listdir(my_local_model_path1) if os.path.isfile(os.path.join(my_local_model_path1, f)) and f not in client_processed_model1]
        client_model_list2 = [f for f in os.listdir(my_local_model_path2) if os.path.isfile(os.path.join(my_local_model_path2, f)) and f not in client_processed_model2]
        client_model_list3 = [f for f in os.listdir(my_local_model_path3) if os.path.isfile(os.path.join(my_local_model_path3, f)) and f not in client_processed_model3]
        global_model_list = [f for f in os.listdir(my_global_model_path) if os.path.isfile(os.path.join(my_global_model_path, f)) and f not in global_processed_model] 

        logging.info('client_model_list1: {}'.format(client_model_list1))
        logging.info('sorted client_model_list1: {}'.format(sorted(client_model_list1, key=natural_keys)))

        if len(client_model_list1) == 0 or len(client_model_list2) == 0 or len(client_model_list3) == 0 or len(global_model_list) == 0:
            logging.info('No more model to process!')
            time.sleep(180)
            continueCount += 1
            if continueCount <= 5:
                continue
            else:
                break

        continueCount = 0        
        local_acc_all, local_loss_all = [], []
        client_model_all_rounds = [None for i in range(num_clients)] # [None, None, None]
        client_model_selection_matrix = [False for i in range(num_clients)] # [False, False, False]

        current_directory = os.getcwd() #'/mnt/data/home/juniarto/shapleyserver'
       
        #for model1, model2, model3 in zip (sorted(client_model_list1, key=natural_keys), sorted(client_model_list2, key=natural_keys), sorted(client_model_list3, key=natural_keys)):
        for (j, (model1, model2, model3, global_avg_model)) in enumerate(zip (sorted(client_model_list1, key=natural_keys), sorted(client_model_list2, key=natural_keys), sorted(client_model_list3, key=natural_keys),sorted(global_model_list, key=natural_keys))):
            logging.info("**********************************************************************************")
            logging.info("NEW EPOCH")
            logging.info("EPOCH NO: {}".format(j+globalEpochCounter))
           
            filePath_1 = os.path.join(my_local_model_path1, model1)
            filePath_2 = os.path.join(my_local_model_path2, model2)
            filePath_3 = os.path.join(my_local_model_path3, model3)
            filePath_global = os.path.join(my_global_model_path, global_avg_model)  
            print('File path: ', filePath_1)
            filePaths = [filePath_1, filePath_2, filePath_3]
            client_models = [client_model_1, client_model_2, client_model_3]
            logging.info("**********************************************************************************")

            if(checkLocalTrainingModelExist(filePath_1) and checkLocalTrainingModelExist(filePath_2) and checkLocalTrainingModelExist(filePath_3)):
                logging.info('All Local Training Model exists!')
                #LOAD CLIENT MODEL HERE
                for(i, (filePath, client_model)) in enumerate(zip(filePaths, client_models)):
                    logging.info('i: {}'.format(i))
                    ckpt = th.load(filePath)
                    #print_trainable_parameters(ckpt['state_dict'])
                    #print(client_model)
                
                    #print(ckpt['state_dict'])
                    client_model.load_state_dict(ckpt['state_dict'])
                    #print(client_model)
                    
                
                
                    logging.info('Model loaded!')
                    accuracy, loss = evaluation(args, client_model, valid_loader) #server valid_loader
                    logging.info('Accuracy: {}'.format(accuracy))
                    logging.info('Loss: {}'.format(loss))
                    local_acc_all.append(accuracy)
                    local_loss_all.append(loss)

                    client_model_all_rounds[i] = get_difference_between_network_weights(client_model, init_global_model)
                    client_model_selection_matrix[i] = True

                logging.info("=====================================================================================================")
                logging.info("Finish All Client Local Training")
                logging.info('Local accuracy all: {}'.format(local_acc_all)) #[0.21171171171171171, 0.21171171171171171, 0.21171171171171171]
                logging.info('Local loss all: {}'.format(local_loss_all)) #[1.3890263806592236, 1.3890264081525372, 1.3890263944058805]
                #print('Client model all rounds: ', client_model_all_rounds) #very long to print out
                logging.info('Client model selection matrix: {}'.format(client_model_selection_matrix)) #[True, True, True]

                acc_session_all.append(local_acc_all)
                loss_session_all.append(local_loss_all)
                local_acc_all, local_loss_all = [], []
                logging.info('Acc session all: {}'.format(acc_session_all))
                logging.info('Loss session all: {}'.format(loss_session_all))
                logging.info("=====================================================================================================")

                """ 
                # create clients
                clients_all = [ClientBase(id, args, init_global_model, dataset)
                            for id in range(num_clients)] 

                # create the server
                server = ServerBase(args, init_global_model, clients_all,None, valid_loader, None) 
                """
                
                game = Game(clients_all, 
                            server, 
                            init_global_model, 
                            client_model_all_rounds, 
                            client_model_selection_matrix,
                            previous_utility,
                            utility_dim, 
                            args
                            )
                logger = None
                shapley_value = call_shapley_computation_method(args,game, logger) 
                logging.info('Shapley value for first local training: {}'.format(shapley_value))
                #[
                # {0: 0.004504504504504499, 1: 0.004504504504504499, 2: 0.004504504504504499}, 
                # {0: -0.009677513744478625, 1: -0.009677513859185029, 2: -0.009677506638718766}
                #]
                for i in range(utility_dim):
                    shapley_value_all_rounds[i].append(shapley_value[i])
                    shapley_value_sum[i] = {k: shapley_value_sum[i][k] + shapley_value[i][k] for k in shapley_value_sum[i].keys()}
                logging.info('Shapley value all rounds: {}'.format(shapley_value_all_rounds))
                logging.info('Shapley value sum: {}'.format(shapley_value_sum))
                updateShapleyDb(shapley_value_all_rounds, session_id, j+globalEpochCounter)
                shapley_session_all.append(shapley_value)
                logging.info('Shapley session all: {}'.format(shapley_session_all))
                client_processed_model1.append(model1)
                client_processed_model2.append(model2)
                client_processed_model3.append(model3)
        



        


    
            #Get Global Model
            #my_global_model = th.load("/mnt/data/home/juniarto/FromRenuga/global/ViT_epoch_50.pth.tar")
            my_global_model = th.load(filePath_global)
            server.global_model.load_state_dict(my_global_model['state_dict'])
            logging.info('Global Model loaded!')

            fed_valid_acc, fed_valid_loss = evaluation(args, server.global_model, valid_loader)
            logging.info('Global Accuracy: {}'.format(fed_valid_acc))
            logging.info('Global Loss: {}'.format(fed_valid_loss))
            previous_utility[0] = fed_valid_acc
            previous_utility[1] = fed_valid_loss

            """ 
            if (j == 3): #Break after 5 epochs 
                logging.info('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
                logging.info('BREAKING THE LOOP')
                break  
            """
        #if break_all:
        j = j + 1
        globalEpochCounter = globalEpochCounter + j
        for key in utility_map:
            shapley_df = pd.DataFrame(shapley_value_all_rounds[key])
            shapley_df['shapley_value_sum'] = shapley_df[list(shapley_value_all_rounds[key][0].keys())].sum(axis=1)
            shapley_df = shapley_df.cumsum(axis=0)
            logging.info('Shapley df: {}'.format(shapley_df))
        #break
    return shapley_value_all_rounds, shapley_value_sum

def checkLocalTrainingModelExist(filepath):
    def is_file_locked(filepath):
        """ 
        Check if a file is locked by trying to open it in exclusive mode without creating it.
        """
        try:
            fd = os.open(filepath, os.O_RDWR | os.O_EXCL)
            os.close(fd)
            return False
        except OSError as e:
            if e.errno == errno.EEXIST:
                return False  # File exists and is not locked
            return True  # File is locked

    def wait_for_file(filepath):
        """Wait until the file is no longer locked."""
        logging.info('inside wait_for_file: {}'.format(filepath))
        while True:
            if os.path.exists(filepath) and not is_file_locked(filepath):
                return True
            logging.info('Waiting for the file to be unlocked...')
            time.sleep(1)

    if(wait_for_file(filepath)):
        return True

def updateShapleyDb(data, session_id, epoch, r=r):
    #Update Shapley value to Redis
    # Calculate cumulative sum of accuracy and loss for each user
    cumsum_results = {}
    for user_id in data[0][0].keys():  # assuming all dicts have the same user keys
        cumsum_results[user_id] = {
            'accuracy_cumsum': np.cumsum([epoch[user_id] for epoch in data[0]]),
            'loss_cumsum': np.cumsum([epoch[user_id] for epoch in data[1]])
        }
    logging.info('Cumsum results: {}'.format(cumsum_results))
    '''
        {
         0: {'accuracy_cumsum': array([0.05945946, 0.27117117, 0.26666667, 0.26216216]), 
            'loss_cumsum': array([0.47168497, 0.15337227, 0.17599531, 0.20702664])}, 
         1: {'accuracy_cumsum': array([0.05945946, 0.27803066, 0.27352616, 0.26902165]), 
             'loss_cumsum': array([0.47168497, 0.15539408, 0.1780171 , 0.20904844])}, 
         2: {'accuracy_cumsum': array([0.05945946, 0.27744868, 0.27294418, 0.26843967]), 
             'loss_cumsum': array([0.47168497, 0.15688913, 0.17951217, 0.21054354])}
            }
    '''
    # Get the cumulative sum of accuracy for each party for the current epoch
    accuracy_cumsum_party = []
    for x in range(3):
        accuracy_cumsum_party.append(cumsum_results[x]['accuracy_cumsum'].tolist())    
        #accuracy_cumsum_party.append(cumsum_results[x]['accuracy_cumsum'][epoch])
        #wrong because last values is epoch + 1
    logging.info('Accuracy cumsum party: {}'.format(accuracy_cumsum_party))



    #Get Session Info from Redis
    #parties = (r.execute_command('JSON.GET', session_id, '.session_1.parties')).decode('utf-8')
    parties = (r.execute_command('JSON.GET', session_id, '.{}.parties'.format(session_id)))
    print('Parties: ', parties)
    parties = json.loads(parties)
    
    for j in range(3):
        #index = next((k for k, party in enumerate(parties) if party['id'] == j), None)
        index = next((k for k, party in enumerate(parties) if party['id'] == myUserMap[j]), None)
        if index is not None:
            #r.execute_command('JSON.ARRAPPEND', session_id, '.{}.parties[{}].shapley_values'.format(session_id, index), accuracy_cumsum_party[j])
            r.execute_command('JSON.SET', session_id, '.{}.parties[{}].shapley_values'.format(session_id, index), json.dumps(accuracy_cumsum_party[j]))

def initiateShapleyDb(session_id, r=r):
    #Create Shapley value key in Redis
    try:
        data = {
            session_id: {
                "parties": [
                        {"id": myUserMap[0], "shapley_values": []},
                        {"id": myUserMap[1], "shapley_values": []},
                        {"id": myUserMap[2], "shapley_values": []}
                ]
            }}
        r.execute_command('JSON.SET', session_id, '.', json.dumps(data))
    except Exception as e:
        logging.error('Error in initiating Shapley DB: {}'.format(e))
        raise

from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    logging.info('Inside count_parameters')    
    logging.info(table)
    logging.info("Total Trainable Params: {}".format(total_params))
    return total_params
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    """ print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    ) """
    logging.info("trainable params: {} || all params: {} || trainable%: {:.2f}".format(trainable_params, all_param, 100 * trainable_params / all_param))    

def start():
    dataset = getOCTData2()

    #device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    '''
    init_global_model = inception_network().to(device)
    client_model_1 = inception_network().to(device)
    client_model_2 = inception_network().to(device)
    client_model_3 = inception_network().to(device)
    '''
    pretrained_checkpoint  = "google/vit-base-patch16-224-in21k"
    # Load the image processor
    vit_processor = ViTImageProcessor.from_pretrained(pretrained_checkpoint)
    vit_processor.do_rescale=False
        
    #Load the pretrained model
    vit_encoder = ViTModel.from_pretrained(pretrained_checkpoint)
    vit_model = ViTForImageClassification.from_pretrained(pretrained_checkpoint)
    vit_model.config.num_labels = 4  # Set the number of output classes
    vit_model.classifier = torch.nn.Linear(vit_model.config.hidden_size, vit_model.config.num_labels)
    #init_global_model = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', pretrained=True).to(device)
    model=vit_model
    print_trainable_parameters(model)
    logging.info("ViT Model")
    count_parameters(model)
    #print(model)
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=16,lora_alpha=8,target_modules=["query", "value"],lora_dropout=0.05,bias="none",modules_to_save=["classifier"],)
    lora_vit_model = get_peft_model(vit_model, lora_config)
    print_trainable_parameters(lora_vit_model)
    model=lora_vit_model
    logging.info("ViT+LoRa")
    count_parameters(model)
    #print(model)
    #print_trainable_parameters(model)
    #model = nn.DataParallel(model,device_ids = [0,1])
     
    init_global_model =model.to(device)
    client_model_1= model.to(device)
    client_model_2=model.to(device)
    client_model_3=model.to(device)
    
    logging.info('Length of dataset: {}'.format(len(dataset)))

    """ for (i, sample) in enumerate(dataset):
        print(i, sample['image'].shape, sample['label']) """

    first_sample = dataset[0]
    image = first_sample['image']
    label = first_sample['label']
    image_name = first_sample['image_name']
    logging.info('Image shape: {} '.format(image.shape))
    logging.info('Label: {}'.format(label))
    logging.info('Name: {}'.format(image_name))
    try:
        initiateShapleyDb(session_id, r=r)
    except Exception as e:
        logging.error('Error in initiating Shapley DB: {}'.format(e))
        sys.exit(1)
        
    shapley_value_all_rounds, shapley_value_sum = getInitialShapleyValue(dataset, init_global_model, client_model_1, client_model_2, client_model_3)
  

    #LOAD PRETRAINED MODEL
    """ print('Loading pretrained model...')
    snapshot_fname='/mnt/data/home/juniarto/inception-model/inception_epoch_9.pth.tar'
    ckpt = th.load(snapshot_fname)
    init_global_model.load_state_dict(ckpt['state_dict'])
    print('Model loaded!')  """

    """ args=''
    valid_loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=1)
    fed_valid_acc, fed_valid_loss = evaluation(args, init_global_model, valid_loader)
    print('Accuracy: ', fed_valid_acc)
    print('Loss: ', fed_valid_loss) """

    #data_set = getOCTData()
    #args=''
    #valid_loader = DataLoader(data_set['train_data'], batch_size=128, shuffle=False, num_workers=4)
    #fed_valid_acc, fed_valid_loss = evaluation(args, init_global_model, valid_loader)
    
    #data_set, data_info = get_dataset(dataset='cifar10')
    #valid_loader = DataLoader(data_set['train_data'], batch_size=128, shuffle=False, num_workers=4)
    #fed_valid_acc, fed_valid_loss = evaluation(args, init_global_model, valid_loader)
    #Eror: AttributeError: 'XrayDataLoader' object has no attribute 'dataset'
    logging.info('Hello World!')

if __name__ == '__main__':
    start()
