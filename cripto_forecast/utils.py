import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal


def series_train_test_split(X,Y,faulty_indices,test_ratio = 0.8):
    total_len = len(X)
    train_length = int(test_ratio * total_len)
    X_train = X[:train_length]
    X_test = X[train_length:]
    Y_train = Y[:train_length]
    Y_test = Y[train_length:]
    faulty_indices_train = []
    faulty_indices_test = []
    for indice in faulty_indices:
        if indice < train_length:
            faulty_indices_train.append(indice)
        else: 
            faulty_indices_test.append(indice-train_length)
    return X_train, Y_train, X_test, Y_test, faulty_indices_train, faulty_indices_test

class DataTransformer:
    def __init__(self,std_series):

        # self.gaussian = Normal(0.0, 0.5)

        self.transforms = [
            # self.dropout,
            self.add_gaussian_noise,
            # self.dropout_and_noise,
            self.chunk_dropout,
            self.chunk_copy,
            self.chunk_swap,
            # self.alternate_dropout,
            self.channel_dropout,
            self.identity,
        ]

        self.dropout_p = 0.2
        self.chunk_length = 0.2

    def transform(self, x):
        def _fn(x):
            return np.random.choice(self.transforms)(x)

        return _fn(x)

    def add_gaussian_noise(self, x):
        # x = x + self.gaussian.sample(x.shape)
        # return x
        for i in range(x.shape[1]):
            noise = np.random.normal(0,NOISE_LEVEL*std_series[i]/100,x.shape[0])
            x[:,i] = x[:,i] + noise
        return x

    def dropout_and_noise(self, x):
        x = self.add_gaussian_noise(x)
        return self.transforms[0](x)

    def chunk_dropout(self, x):
        chunk_size = int(len(x) * self.chunk_length)
        chunk_start = np.random.randint(0, len(x) - chunk_size)
        x[chunk_start: chunk_start + chunk_size] = 0
        return x

    def chunk_copy(self, x):
        chunk_length = int(len(x) * self.chunk_length)
        chunk_start = np.random.randint(chunk_length * 2,
                                        len(x) - chunk_length * 2)
        chunk_end = chunk_start + chunk_length
        chunk = x[chunk_start: chunk_end].copy()

        direction = np.random.choice(['left', 'right'])
        if direction == 'left':
            start = chunk_start - chunk_length
            end = chunk_start
        else:
            start = chunk_end
            end = chunk_end + chunk_length
        x[start: end] = chunk
        return x

    def chunk_swap(self, x):
        # get a random chunk of the sequence
        chunk_length = int(len(x) * self.chunk_length)
        chunk_start = np.random.randint(chunk_length * 2,
                                        len(x) - chunk_length * 2)

        chunk_end = chunk_start + chunk_length
        chunk = x[chunk_start: chunk_end]

        direction = np.random.choice(['left', 'right'])
        if direction == 'left':
            start = chunk_start - chunk_length
            end = chunk_start
        else:
            start = chunk_end
            end = chunk_end + chunk_length

        # swap
        tmp = x[start: end].copy()
        x[start: end] = chunk
        x[chunk_start: chunk_end] = tmp
        return x

    def dropout(self, x):
        points_to_dropout = np.argwhere(
            np.random.choice([True, False], size=len(x),
                             p=[self.dropout_p, 1 - self.dropout_p]))
        x[points_to_dropout.reshape(-1)] = 0
        return x

    def alternate_dropout(self, x):
        x[::2] = 0
        return x

    def channel_dropout(self, x):
        num_channels = x.shape[1]
        channel = np.random.choice(list(range(num_channels)))
        x[:, channel] = 0
        return x

    def identity(self, x):
        return x

class TimeseriesDataset(torch.utils.data.Dataset):   
    def __init__(self, X, y, faulty_indices, std_series, PASOS_FUTURO, NOISE_LEVEL, REDUCTION_STEP , seq_len=10, is_train = 1):
        self.X = X
        self.y = y
        self.seq_len = seq_len
        self.faulty_indices = faulty_indices
        self.transformer = DataTransformer(std_series)
        self.is_train = is_train
        self.PASOS_FUTURO = PASOS_FUTURO
        self.NOISE_LEVEL = NOISE_LEVEL
        self.REDUCTION_STEP = REDUCTION_STEP
        self.diccionario_indices = self.generate_dict()

    def generate_dict(self):
        # Numero total de instancias en el Dataset
        numero_total = len(self.X)

        # Cantidad de pasos necesarios para definir una instancia
        ventanita = np.ones(self.seq_len + self.PASOS_FUTURO)

        # Definimos las regiones donde no hay valores faltantes en la ventana
        fallas = np.zeros(numero_total)
        for i in self.faulty_indices:
            fallas[i] = 1
        region_fallas = np.convolve(fallas, ventanita, mode='valid')
        len_convolve = len(region_fallas) # numero_total - len(ventanita) + 1

        # Contamos las instancias validas
        dataset_len = len_convolve - (region_fallas>0).sum()

        # Definimos un diccionario para los indices validos
        indices_fallas = np.arange(len_convolve)
        diccionario_indices = indices_fallas[region_fallas==0]

        return diccionario_indices

    def __len__(self):
        dataset_len = len(self.diccionario_indices)
        return dataset_len

    def __getitem__(self, index):
        dict_index = self.diccionario_indices[index]
        instance = self.X[dict_index:dict_index+self.seq_len]
        # for i in range(X.shape[1]):
        #     noise = np.random.normal(0,self.NOISE_LEVEL*std_series[i]/100,self.seq_len)
        #     instance[:,i] = instance[:,i] + noise
        target = self.y[dict_index+self.seq_len]

        instance = instance[len(instance)%self.REDUCTION_STEP:]
        instance = np.add.reduceat(instance, np.arange(0, len(instance), self.REDUCTION_STEP),axis = 0)
        if self.is_train == 1:
            instance = self.transformer.transform(instance)
            scaling_factor = np.random.uniform(0.5, 1.5)
            instance = instance * scaling_factor
            if REGRESSION == 1:
                target = target * scaling_factor
                
        instance = np.transpose(instance, (1, 0))       
        return (instance,target)
    
class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        # the first conv
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        # the first conv
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        # the second conv
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        # if downsample, also downsample identity
        if self.downsample:
            identity = self.max_pool(identity)
            
        # if expand channel, also pad zeros to identity
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        # shortcut
        out += identity

        return out
    
class ResNet1D(nn.Module):
    """
    
    Input:
        X: (n_samples, n_channel, n_length)
        Y: (n_samples)
        
    Output:
        out: (n_samples)
        
    Pararmetes:
        in_channels: dim of input, the same as n_channel
        base_filters: number of filters in the first several Conv layer, it will double at every 4 layers
        kernel_size: width of kernel
        stride: stride of kernel moving
        groups: set larget to 1 as ResNeXt
        n_block: number of blocks
        n_classes: number of classes
        
    """

    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=2, increasefilter_gap=4, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap # 2 for base model
        self.increasefilter_gap = increasefilter_gap # 4 for base model

        # first block
        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
                
        # residual blocks
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            # is_first_block
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
            # downsample at every self.downsample_gap blocks
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            # in_channels and out_channels
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:
                # increase filters at every self.increasefilter_gap blocks
                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        # final prediction
        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)
        # self.do = nn.Dropout(p=0.5)
        self.dense = nn.Linear(out_channels, n_classes)
        # self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        out = x
        
        # first conv
        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)
        out = self.first_block_relu(out)
        
        # residual blocks, every block has two conv
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)
            if self.verbose:
                print(out.shape)

        # final prediction
        if self.use_bn:
            out = self.final_bn(out)
        out = self.final_relu(out)
        out = out.mean(-1)
        if self.verbose:
            print('final pooling', out.shape)
        # out = self.do(out)
        out = self.dense(out)
        if self.verbose:
            print('dense', out.shape)
        # out = self.softmax(out)
        if self.verbose:
            print('softmax', out.shape)
        
        return out
