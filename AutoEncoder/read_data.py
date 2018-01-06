import torch
from mydataset import MyDataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.autograd import Variable
BATCH_SIZE = 5
EMBDDING_DIM = 116
LR = 1e-3
filename = "res-0106-rnn.txt"
def read_data_vec(filename):
    word_vec_dict = {}
    word_dict = {}
    f = open(filename, "r+")
    for line in f.readlines():
        data_pair = line.split("*")
        word = data_pair[0]
        if word not in word_dict:
            word_dict[len(word_dict)] = word
        vec_str = data_pair[1]
        tmp = vec_str.replace('\t', '').strip('\n').strip('[').strip(']').split(', ')
        vec = [round(float(i), 5) for i in tmp]
        # for i in tmp:
        #    print(float(i))
        word_vec_dict[word] = vec
    return word_dict,word_vec_dict


word_dict,word_vec_dict = read_data_vec(filename)
#vecs = list(word_vec_dict.values())
vecs = [torch.FloatTensor(i) for i in word_vec_dict.values()]
dataset = MyDataset(vecs,vecs)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)

test_data = vecs[:10]


class AutoEncoder(nn.Module):

    def __init__(self,embdding_size):
        super(AutoEncoder,self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(embdding_size, 58),
            nn.Tanh(),
            nn.Linear(58, 29),
            nn.Tanh(),
            nn.Linear(29, 15),
            #nn.Tanh(),
            #nn.Linear(12, 3),  # compress to 3 features which can be visualized in plt
        )
        self.decoder = nn.Sequential(
            nn.Linear(15, 29),
            nn.Tanh(),
            nn.Linear(29, 58),
            nn.Tanh(),
            nn.Linear(58, embdding_size),
        )

    def forward(self,x):
        encoder_data = self.encoder(x)
        decoder_data = self.decoder(encoder_data)
        return encoder_data,decoder_data

autoencoder = AutoEncoder(EMBDDING_DIM)
criterion = nn.MSELoss()
optimizer = optim.Adam(autoencoder.parameters(),lr=LR)


for epoch in range(100):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    for _,(x,y) in enumerate(train_loader):
        x = Variable(x)
        y = Variable(y)

        encoder_data, decoder_data = autoencoder(x)
        running_loss = criterion(decoder_data,y)

        optimizer.zero_grad()
        running_loss.backward()
        optimizer.step()

    print('Loss: {}'.format(running_loss))

for i,test_case in enumerate(test_data):
    test_var = Variable(test_case)
    encoder, decoder = autoencoder(test_var)
    print(word_dict[i])
    print("Origin data: ", word_vec_dict[word_dict[i]])
    print("Encoder data: ",list(encoder.data.numpy()))
    print("Deconder data",list(decoder.data.numpy()))
    print('*'*10)








