import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.functional as F
import os
import numpy as np

carla = [
   [70 ,70 ,70],
   [150 , 60,  45],
   [180 ,130 , 70],
   [232 , 35 ,244],
   [ 35 ,142 ,107],
   [100 ,170 ,145],
   [160 ,190 ,110],
   [153 ,153 ,153],
   [80 ,90 ,55],
   [ 50 ,120 ,170],
   [128 , 64 ,128],
   [ 50 ,234 ,157],
   [142 ,  0  , 0],
   [  0 ,220 ,220],
   [ 30 ,170 ,250],
   [156 ,102 ,102],
   [ 40 , 40 ,100],
   [81  ,0 ,81],
   [140 ,150 ,230],
   [100 ,100 ,150]
]
carla = np.array(carla)


device = "cuda:0"

Loss = nn.CrossEntropyLoss()

class Embedder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                       log_sampling=True, include_input=True,
                       periodic_fns=(torch.sin, torch.cos)):
        '''
        :param input_dim: dimension of input to be embedded
        :param max_freq_log2: log2 of max freq; min freq is 1 by default
        :param N_freqs: number of frequency bands
        :param log_sampling: if True, frequency bands are linerly sampled in log-space
        :param include_input: if True, raw input is included in the embedding
        :param periodic_fns: periodic functions used to embed input
        '''
        super().__init__()
        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.out_dim = 0
        if self.include_input:
            self.out_dim += self.input_dim

        self.out_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input):
        '''
        :param input: tensor of shape [..., self.input_dim]
        :return: tensor of shape [..., self.out_dim]
        '''
        assert (input.shape[-1] == self.input_dim)

        out = []
        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            freq = self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))
        out = torch.cat(out, dim=-1)

        assert (out.shape[-1] == self.out_dim)
        return out

class my_dataset(Dataset):
    """
    The dataset calss for pixel postision
    :param
    img_path: the gt of segmantation, size:512 512 n_class
    mask_path: the valid position to train and the missing position to complte, size: 512 512
    """
    def __init__(self,img_path, mask_path):
        super(my_dataset,self).__init__()

        self.img_path = img_path
        self.mask_path = mask_path

        self.img = np.array(Image.open(self.img_path))
        self.mask = np.array(Image.open(self.mask_path))
        assert (self.img.shape[0] == self.mask.shape[0]) and (self.img.shape[1] == self.mask.shape[1]), "Image shape is not equal to mask shape"
        assert self.mask.ndim ==2 ,"the ndim of mask need to be two"
        self.shape = self.img.shape
        self.position = self.mask_to_position()


    def mask_to_position(self,):
        """
        obain the pixel position in Image plane
        """
        shape = self.shape
        x, y = np.meshgrid(np.arange(shape[0], dtype = np.float32),
                           np.arange(shape[1], dtype=np.float32))
        position = np.stack([y,x], 2)  ###dont be sure which is colum or row... row = x??
        return position

    def __getitem__(self, index):
        """
        :param index:
        :return: the valid position and valid gt corresponding to the position

        where mask ==1  the position is valid  need to train
        """
        shape = self.shape
        ## Now maks is wrong
        mask_stretch = self.mask.reshape(shape[0]*shape[1])
        img_stretch = self.img.reshape(shape[0]*shape[1],-1)
        # print(np.size(mask_stretch))
        position_stretch = self.position.reshape(shape[0]*shape[1],2)
        position_valid = np.where(mask_stretch==1, True, False)
        # print(np.size(position_valid))
        position_stretch_valid = position_stretch[position_valid==True,:]
        img_stretch_valid = img_stretch[position_valid==True,:]
        ###from numpy to tensor
        position_stretch_valid = torch.from_numpy(position_stretch_valid).float().to(device= device)
        img_stretch_valid = torch.from_numpy(img_stretch_valid).int().to(device= device)
        # print(position_stretch_valid.shape)


        return {
                "position":position_stretch_valid[index,:],
                "class": img_stretch_valid[index,:]
                }


    def __len__(self):

        return np.sum(np.where(self.mask == 1 ,1,0))


class my_Net(nn.Module):
    """
    :param
    n_class
    """
    def __init__(self, n_class):
        super(my_Net,self).__init__()

        self.layer1 = nn.Linear(42,100)
        self.layer2 = nn.Linear(100,60)
        self.layer3 = nn.Linear(60,20)
        self.layer4 = nn.Linear(20,n_class)
        self.relu = nn.ReLU()
        self.sigmod = nn.Sigmoid()
        self.embedder = Embedder(input_dim=2,
                 max_freq_log2=10 - 1,
                 N_freqs=10)

    def forward(self, x):
        x = self.embedder(x)   ####  C 42

        # print(x.shape)
        x = self.layer1(x)
        x = self.relu(x)
        # print(x.shape)
        x = self.layer2(x)
        x = self.relu(x)

        x = self.layer3(x)
        x = self.relu(x)

        x = self.layer4(x)
        x = self.relu(x)

        # x = self.sigmod(x)

        return x

def train(img_path, mask_path, n_class, checkpoints_dir, continue_traning = False):
    if not os.path.exists(checkpoints_dir):
        os.mkdir(checkpoints_dir)

    #### Now the mask image is not correct
    dataset = my_dataset(img_path, mask_path)
    train_dataloader = DataLoader(dataset = dataset, batch_size=1000,shuffle= True)

    net = my_Net(n_class).to(device = device)
    net.train()
    optimizer = optim.Adam(net.parameters(), lr = 1e-3)

    loss_all = []
    loss_epoch = []
    total_epoch = 100
    start = 0
    state = OrderedDict()

    if continue_traning:
        to_load = torch.load("./checkpoints/checkpoint_8.pth")
        start = to_load["epoch"]
        net.load_state_dict(to_load["net"], strict=False)
    print(len(train_dataloader))
    for i in range(start, total_epoch):
        for index, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            result = net(batch["position"].unsqueeze(1))   ### b 1  2
            gt = batch["class"][...,0]    ###100 3-> 100
            result = result.permute(0,2,1)[...,0]  ###b 1 20 -> b 20
            # print(result.shape, gt.shape)
            loss = Loss(result,gt.long())    ###B C , B
            loss.backward()
            loss_epoch.append(loss.item())
            # print(result.shape)
            # result = F.softmax(result, dim=1)
            # result = torch.argmax(result, dim=1)
            # print(gt)
            print("epoch_{}/iter_{}: {}".format(i, index, loss.item()))
            optimizer.step()

        loss_all.append(np.mean(loss_epoch))
        print(np.mean(loss_epoch))
        loss_epoch.clear()
        print("saving...")
        state = {"net" :net.state_dict(), 'optimizer':optimizer.state_dict(), "epoch":i}
        torch.save(state, os.path.join(checkpoints_dir,"checkpoint_{}.pth".format(i)))
        plt.cla()
        plt.plot(range(0, len(loss_all)), loss_all)
        plt.savefig("./loss_all.png")
def test(n_class, checkpoint):
    net = my_Net(n_class).to(device=device)
    net.eval()
    to_load = torch.load(os.path.join("./checkpoints",checkpoint))
    # start = to_load["epoch"]
    net.load_state_dict(to_load["net"], strict=False)

    ##no need dataloader, process all position
    # dataset = my_dataset("./img.png", "./mask.png")
    # test_dataloader = DataLoader(dataset=dataset, batch_size=100)

    shape = (512,512)
    x, y = np.meshgrid(np.arange(shape[0], dtype=np.float32),
                       np.arange(shape[1], dtype=np.float32))
    position = np.stack([y, x], 2)
    position = torch.from_numpy(position).float().to(device=device).reshape([-1,2]).unsqueeze(1)
    print(position.shape)
    results = net(position)
    print(results.shape)
    img = F.softmax(results, dim=-1)
    img = torch.argmax(img, dim=-1)
    img = torch.reshape(img,(512,512)).cpu().numpy()
    result_color = np.ones((512,512,3)).astype(np.uint8)
    for x in range(512):
        for y in range(512):
            result_color[x,y,:] = carla[img[x,y],:]

    Image.fromarray(result_color).convert("L").save("./result_color.png")



    print(img.shape)


if __name__ == "__main__":
    # dataset = my_dataset("./imgs/img.png", "./imgs/row_mask.png")
    # print(len(dataset))
    # print(dataset[0])
    #train = DataLoader(dataset = dataset, batch_size=100,shuffle= True)
    # for i, data in enumerate(train):
    #     print(i,":", data.shape)

    checkpoints_dir = "./checkpoints"
    n_class = 20
    # train("./imgs/img.png", "./imgs/row_mask.png", n_class, checkpoints_dir, continue_traning=False)
    test(n_class, checkpoint = "checkpoint_6.pth")


