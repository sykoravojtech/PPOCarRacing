import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import random
import copy

class FCNet(nn.Module):
    def __init__(self):
        super(FCNet, self).__init__()
        self.fc = nn.Linear(in_features=28 * 28,
                            out_features=1000)
        self.fc2 = nn.Linear(1000, 1000)
        self.fc3 = nn.Linear(1000, 10)
        

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=10,
                              kernel_size=3,
                              stride=2,
                              padding=1)
        self.fc = nn.Linear(in_features=28 * 28 * 10 // (2 * 2),
                            out_features=10)

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        output = F.log_softmax(x, dim=1)
        return output


class MyNet(nn.Module):
    """
    Experiment with all possible settings mentioned in the CW page
    """    
    def __init__(self):
        super(MyNet, self).__init__() 
               
        self.C1 = myConvBlock(1, 64, kernel_size = 3, stride = 2, padding = 1)
        self.MP1 = nn.MaxPool2d(2)
        
        self.C2 = myConvBlock(64, 128,3,1,1)
        self.C3 = myConvBlock(128, 128,3,1,1)
        self.C4 = myConvBlock(128, 128,3,1,1)
        self.C5 = myConvBlock(128 + 128, 128,3,1,1) #skip 128 C2, 128 C4
        self.MP2 = nn.MaxPool2d(2)
        
        self.C6 = myConvBlock(128, 256,3,1,1)
        self.C7 = myConvBlock(256, 256,3,1,1)
        self.C8 = myConvBlock(256, 256,3,1,1)
        self.C9 = myConvBlock(256 + 256, 256,3,1,1) #skip 256 C6, 256 C8
        self.MP3 = nn.MaxPool2d(2)
        
        self.C10 = myConvBlock(256, 512,3,1,1)
        self.C11 = myConvBlock(512, 512,3,1,1)
        self.C12 = myConvBlock(512, 512,3,1,1)
        self.C13 = myConvBlock(512 + 512, 512,3,1,1) #skip 512 C10, 512 C12
        # self.MP4 = nn.MaxPool2d(2)
        
        self.flat = nn.Flatten()
        
        self.L1 = myLinearBlock(512,1024)
        self.L2 = myLinearBlock(1024,512)
        self.L3 = myLinearBlock(512,10)
        
        for layer in self.modules():
            if type(layer) == torch.nn.Conv2d or type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.C1(x)
        x = self.MP1(x)
        x = self.C2(x)
        
        skip1 = x
        x = self.C3(x)
        x = self.C4(x)
        x = torch.cat((x, skip1), dim=1)
        x = self.C5(x)
        x = self.MP2(x)
        
        x = self.C6(x)
        skip2 = x
        x = self.C7(x)
        x = self.C8(x)
        x = torch.cat((x, skip2), dim=1)
        x = self.C9(x)
        x = self.MP3(x)
        
        x = self.C10(x)
        skip3 = x
        x = self.C11(x)
        x = self.C12(x)
        x = torch.cat((x, skip3), dim=1)
        x = self.C13(x)
        # x = self.MP4(x)
        
        x = self.flat(x)
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        
        out = F.log_softmax(x, dim=1)
        return out
    
def myConvBlock(inp=1, out=128, kernel_size=3, stride=1, padding=1, drop=0.02):
    return nn.Sequential(
        nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=kernel_size, stride=stride, padding=padding),
        nn.Dropout2d(drop),
        nn.BatchNorm2d(out),
        nn.GELU()
    )

def myConvBlockOfBlocks(inp=1, out=128, kernel_size=3, stride=1, padding=1, drop=0.02, repeat=1):
    layers = []
    for i in range(repeat):
        layers.append(nn.Conv2d(in_channels=inp, out_channels=out, kernel_size=kernel_size, stride=stride, padding=padding))
        layers.append(nn.Dropout2d(drop))
        layers.append(nn.BatchNorm2d(out))
        layers.append(nn.GELU())
        
    return nn.Sequential(*layers)
        
def myLinearBlock(inp, out, drop = 0.04):
    return nn.Sequential(
        nn.Linear(in_features=inp, out_features=out),
        nn.Dropout(drop),
        nn.GELU()
    )

def classify(model, x):
    """
    :param model:    network model object
    :param x:        (batch_sz, 1, 28, 28) tensor - batch of images to classify

    :return labels:  (batch_sz, ) torch tensor with class labels
    """
    softmax_labels = model.forward(x)
    labels = torch.argmax(softmax_labels, dim=1)
    return labels


def get_model_class(_):
    """ Do not change, needed for AE """
    return [MyNet]

class TransformedFashionMNIST(torch.utils.data.Dataset):
    def __init__(self, images):
        super().__init__()
        self.images = images
        self.num_of_transforms = 3
        self.transforms = [
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,)),
            transforms.RandomErasing(p=0.5, value="random"),
            transforms.RandomAutocontrast(),
            transforms.RandomInvert(),
            transforms.ColorJitter(brightness=1),
            transforms.RandomRotation(90),
            transforms.RandomPerspective(),
            transforms.GaussianBlur(kernel_size=5, sigma=(2,5)),
            transforms.RandomAdjustSharpness(5),
        ]
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        x, rest = self.images[index]
        for _ in range(self.num_of_transforms):
            x = random.choice(self.transforms)(x)
        return x, rest

def train(network_cls : nn.Module, lr = 0.01, lr_decay = 0.99, epochs = 40, batch_sz = 64, models_dir = "models", num_workers = 8, load = False):
    print(f"{batch_sz = } {lr = } {lr_decay = } {models_dir = }")
    
    dataset = datasets.FashionMNIST('data', train=True, download=True,
                                    transform=transforms.ToTensor()) #original
    
    # print(f"dataset {dataset}")

    trn_size = int(0.95 * len(dataset))
    val_size = int(0.05 * len(dataset))
    add_size = len(dataset) - trn_size - val_size  # you don't need ADDitional dataset to pass

    trn_dataset, val_dataset, add_dataset = torch.utils.data.random_split(dataset, [trn_size,
                                                                                    val_size,
                                                                                    add_size])
    
    add_dataset = TransformedFashionMNIST(copy.deepcopy(trn_dataset))
    
    trn_add_dataset = torch.utils.data.ConcatDataset([trn_dataset, add_dataset])
    print(f"{len(trn_dataset) = } {len(add_dataset) = } {len(trn_add_dataset) = }")
    
    trn_loader = torch.utils.data.DataLoader(trn_add_dataset,
                                             batch_size=batch_sz,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             pin_memory=True)

    val_augmented_dataset = TransformedFashionMNIST(copy.deepcopy(val_dataset))
    val_final_dataset = torch.utils.data.ConcatDataset([val_dataset, val_augmented_dataset])
    val_loader = torch.utils.data.DataLoader(val_final_dataset,
                                             batch_size=batch_sz,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nn.DataParallel(network_cls.to(device))
    
    if load:
        print("Loading model")
        model.module.load_state_dict(torch.load("./adam/lr0.001_b512/model.pt"))
        models_dir = "testing"

    optimizer = optim.Adam(model.parameters(), lr=lr, amsgrad=True, weight_decay=0.0)
    
    best_accuracy = 0
    
    lr_schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.4, patience=7)
    
    for epoch in range(1, epochs + 1):
        # training
        
        model.train()
        for i_batch, (x, y) in enumerate(trn_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            net_output = model(x)
            loss = F.nll_loss(net_output, y)
            loss.backward()
            optimizer.step()

            if i_batch % 100 == 0:
                print(f'[TRN] Train epoch: {epoch}, batch: {i_batch}\tLoss: {loss.item():.4f}')
                
        # validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                net_output = model(x)

                prediction = classify(model, x)
                correct += prediction.eq(y).sum().item()
        val_accuracy = correct / len(val_loader.dataset)
        lr_schedule.step(val_accuracy)

        torch.save(model.state_dict(), f"{models_dir}/model_last.pt")
        if val_accuracy > best_accuracy:
            print(f'  [VAL] Validation accuracy: {100 * val_accuracy:.2f}% lr={lr} NEW_BEST')
            best_accuracy = val_accuracy
            
            with open(f"{models_dir}/accuracy", "w") as f:
                f.write(f"accuracy {val_accuracy}")
                
            if val_accuracy > 0.95:
                torch.save(model.module.state_dict(), f"{models_dir}/model_best_{100 * val_accuracy:.2f}.pt")
            else:
                torch.save(model.module.state_dict(), f"{models_dir}/model.pt")
        else:
            print(f'  [VAL] Validation accuracy: {100 * val_accuracy:.2f}% lr={lr}')
            
        # lr *= lr_decay

# -------------------------------------------------------------
if __name__ == '__main__':  
    from time import time
    import os
    
    str = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"gpu/cpu = {str}")

    num_of_gpus = torch.cuda.device_count()
    print(f"{num_of_gpus = }")
    
    LR = 0.001
    BATCH_SIZE = 512
    MODELS_DIR = f"adam/lr{LR}_b{BATCH_SIZE}"
    # MODELS_DIR = "testing"
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
    
    train(
        MyNet(),
        lr = LR,
        lr_decay = 0.9999,
        epochs = 5_000_000,
        batch_sz = BATCH_SIZE,
        models_dir = MODELS_DIR,
        num_workers = 2,
        # load = True
        )
    
    # for w in [2,4,6,8,10,12,14,16,18,20]:
    #     start = time()
    #     train(
    #         MyNet(),
    #         learning_rate = 0.001,
    #         epochs = 20,
    #         batch_sz = 2048,
    #         models_dir = "models",
    #         num_workers = w
    #     )
    #     print(f"{w = } took {time()-start} seconds")
  

    
    # PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
    # torchvision/0.13.1-foss-2022a-CUDA-11.7.0
    
    """
    workers 2, 16.96s
    """

# ORIGINAL 8 point brute
"""    
    def __init__(self):
        super(MyNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.025),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.Dropout2d(0.025),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.Linear = nn.Sequential(
            nn.Linear(in_features=12544, out_features=600),
            # nn.Dropout2d(0.25),
            nn.Linear(in_features=600, out_features=120),
            nn.Linear(in_features=120, out_features=10)
        )
        
        

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.Linear(x)
        out = F.log_softmax(x, dim=1)
        return out
"""

"""
19.31
    def __init__(self):
        super(MyNet, self).__init__()        
        self.C1 = myConvBlock(1,32,3,1,1)
        self.C2 = myConvBlock(32,64,3,1,1)
        self.MP1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.C3 = myConvBlock(64,128,3,1,1)
        self.C4 = myConvBlock(128,128,3,1,1)
        self.MP2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.C5 = myConvBlock(128,256,3,1,1)
        self.C6 = myConvBlock(256,256,3,1,1)
        self.MP3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.flat = nn.Flatten()
        
        self.L1 = myLinearBlock(2304,1024)
        self.L2 = myLinearBlock(2048,1024)
        self.L3 = myLinearBlock(1024,512)
        self.L4 = myLinearBlock(512,256)
        self.L5 = myLinearBlock(256,10)
        
        for layer in self.modules():
            if type(layer) == torch.nn.Conv2d or type(layer) == torch.nn.Linear:
                torch.nn.init.xavier_uniform_(layer.weight)

    def forward(self, x):
        x = self.C1(x)
        x = self.C2(x)
        x = self.MP1(x)
        
        x = self.C3(x)
        x = self.C4(x)
        x = self.MP2(x)
        
        x = self.C5(x)
        x = self.C6(x)
        x = self.MP3(x)
        
        x = self.flat(x)
        
        x = self.L1(x)
        x = self.L2(x)
        x = self.L3(x)
        x = self.L4(x)
        x = self.L5(x)
        
        out = F.log_softmax(x, dim=1)
        return out
"""