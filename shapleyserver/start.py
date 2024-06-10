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

load_dotenv()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

my_local_model_path = os.getenv("LOCAL_MODEL_PATH")
my_global_model_path = os.getenv("GLOBAL_MODEL_PATH")
my_validation_dataset = os.getenv("VALIDATION_DATASET")

print('my_validation_dataset: ', my_validation_dataset)

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
    print('Previous utility: ', previous_utility) #[0.21171171171171171, 1.5319562344937712]

    shapley_value_all_rounds = [[] for _ in range(utility_dim)]  #[[], []]
    shapley_value_sum = [{} for _ in range(utility_dim)]         #[{}, {}]

    num_clients=3
  
    # Initial Shapley value
    for i in range(utility_dim):
        shapley_value_all_rounds[i].append({client_id: previous_utility[i] / num_clients for client_id in range(num_clients)})
        shapley_value_sum[i] = shapley_value_all_rounds[i][0]
    
    print('shapley_value_all_rounds: {}'.format(shapley_value_all_rounds))
    """ 
    [
        [{0: 0.059159159159159154, 1: 0.059159159159159154, 2: 0.059159159159159154}], ---> accuracy 
        [{0: 0.5220634746837902, 1: 0.5220634746837902, 2: 0.5220634746837902}]        ---> loss 
    ]
    """
    print('shapley_value_sum: {}'.format(shapley_value_sum))
    """
    [
        {0: 0.059159159159159154, 1: 0.059159159159159154, 2: 0.059159159159159154},    ---> accuracy
        {0: 0.5220634746837902, 1: 0.5220634746837902, 2: 0.5220634746837902}           ---> loss
    ]
    """

    #client local training
    local_acc_all, local_loss_all = [], []
    client_model_all_rounds = [None for i in range(num_clients)] # [None, None, None]
    client_model_selection_matrix = [False for i in range(num_clients)] # [False, False, False]

    current_directory = os.getcwd() #'/mnt/data/home/juniarto/shapleyserver'
    '''
    filePath_1 = os.path.join(current_directory, 'shapleyserver', 'local_training', 'client_1_model','inception_epoch_9.pth.tar')
    filePath_2 = os.path.join(current_directory, 'shapleyserver', 'local_training', 'client_2_model','inception_epoch_9.pth.tar')
    filePath_3 = os.path.join(current_directory, 'shapleyserver', 'local_training', 'client_3_model','inception_epoch_9.pth.tar')
    '''
    filePath_1 = os.path.join(current_directory, 'shapleyserver', 'local_training', 'client_1_model','ViT_epoch_9.pth.tar')
    filePath_2 = os.path.join(current_directory, 'shapleyserver', 'local_training', 'client_2_model','ViT_epoch_9.pth.tar')
    filePath_3 = os.path.join(current_directory, 'shapleyserver', 'local_training', 'client_3_model','ViT_epoch_9.pth.tar')
    print('File path: ', filePath_1)
    filePaths = [filePath_1, filePath_2, filePath_3]
    client_models = [client_model_1, client_model_2, client_model_3]

    if(checkLocalTrainingModelExist(filePath_1) and checkLocalTrainingModelExist(filePath_2) and checkLocalTrainingModelExist(filePath_3)):
        print('All Local Training Model exists!')
        #LOAD CLIENT MODEL HERE
        for(i, (filePath, client_model)) in enumerate(zip(filePaths, client_models)):
            print('i: ', i)
            ckpt = th.load(filePath)
            #print_trainable_parameters(ckpt['state_dict'])
            #print(client_model)
           
            #print(ckpt['state_dict'])
            client_model.load_state_dict(ckpt['state_dict'])
            #print(client_model)
            
           
           
            print('Model loaded!')
            accuracy, loss = evaluation(args, client_model, valid_loader) #server valid_loader
            print('Accuracy: ', accuracy)
            print('Loss: ', loss)
            local_acc_all.append(accuracy)
            local_loss_all.append(loss)

            client_model_all_rounds[i] = get_difference_between_network_weights(client_model, init_global_model)
            client_model_selection_matrix[i] = True


        print('Local accuracy: ', local_acc_all)
        print('Local loss: ', local_loss_all)
        #print('Client model all rounds: ', client_model_all_rounds) #very long to print out
        print('Client model selection matrix: ', client_model_selection_matrix)

        # create clients
        clients_all = [ClientBase(id, args, init_global_model, dataset)
                       for id in range(num_clients)] 

        # create the server
        server = ServerBase(args, init_global_model, clients_all,None, valid_loader, None)
        
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
        print('inside wait_for_file: ', filepath)
        while True:
            if os.path.exists(filepath) and not is_file_locked(filepath):
                return True
            print('Waiting for the file to be unlocked...')
            time.sleep(1)

    if(wait_for_file(filepath)):
        return True
    
from prettytable import PrettyTable
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print("Total Trainable Params: {total_params}")
    return total_params
    
def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )    

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
    print("ViT Model")
    count_parameters(model)
    print(model)
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(r=16,lora_alpha=8,target_modules=["query", "value"],lora_dropout=0.05,bias="none",modules_to_save=["classifier"],)
    lora_vit_model = get_peft_model(vit_model, lora_config)
    print_trainable_parameters(lora_vit_model)
    model=lora_vit_model
    print("ViT+LoRa")
    count_parameters(model)
    print(model)
    print_trainable_parameters(model)
    model = nn.DataParallel(model,device_ids = [0,1])
     
    init_global_model =model.to(device)
    client_model_1= model.to(device)
    client_model_2=model.to(device)
    client_model_3=model.to(device)
    
    print('Length of dataset: ', len(dataset))

    """ for (i, sample) in enumerate(dataset):
        print(i, sample['image'].shape, sample['label']) """

    first_sample = dataset[0]
    image = first_sample['image']
    label = first_sample['label']
    image_name = first_sample['image_name']
    print('Image shape: ', image.shape)
    print('Label: ', label)
    print('Name: ', image_name)

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
    print('Hello World!')

if __name__ == '__main__':
    start()
