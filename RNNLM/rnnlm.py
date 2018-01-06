import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.autograd import Variable
from numpy import *
from torch.utils.data import DataLoader
from mydataset import MyDataset
from data_preprocess import get_data_list
BATCH_SIZE = 5
'''
sentence_set = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()
'''
filename = "Hamlet.txt"
sentence_set = get_data_list(filename)

EMBDDING_DIM = len(set(sentence_set))+1
HIDDEN_UNITS = 200
word_to_ix = {}
for word in sentence_set:
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)
print(word_to_ix)


def make_word_to_ix(word,word_to_ix):
    vec = torch.zeros(EMBDDING_DIM)
    #vec = torch.LongTensor(EMBDDING_DIM,1).zero_()
    if word in word_to_ix:
        vec[word_to_ix[word]] = 1
    else:
        vec[len(word_to_ix)] = 1
    return vec


data_words = []
data_labels = []
for i in range(len(sentence_set) -2):
    word = sentence_set[i]
    label = sentence_set[i+1]
    data_words.append(make_word_to_ix(word,word_to_ix))
    data_labels.append(make_word_to_ix(label,word_to_ix))

dataset = MyDataset(data_words, data_labels)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE,shuffle=True)

'''
for x in enumerate(train_loader):
    print("word_batch------------>\n")
    print(batch[0])
    print("label batch----------->\n")
    print(batch[1])
'''

class RNNModel(nn.Module):

    def __init__(self, embdding_size, hidden_size):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(embdding_size, hidden_size,num_layers=1,nonlinearity='relu')
        self.linear = nn.Linear(hidden_size, embdding_size)

    def forward(self, x, hidden):
        output1, h_n = self.rnn(x, hidden)
        output2 = self.linear(output1)
        log_prob = F.log_softmax(output2)
        return log_prob, h_n


rnnmodel = RNNModel(EMBDDING_DIM, HIDDEN_UNITS)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(rnnmodel.parameters(), lr=1e-3)
'''
#testing
#input_hidden = torch.autograd.Variable(torch.randn(BATCH_SIZE, HIDDEN_UNITS))
#x = torch.autograd.Variable(torch.rand(BATCH_SIZE,EMBDDING_DIM))
#y,_ = rnnmodel(x,input_hidden)
#print(y)
#'''
for epoch in range(3):
    print('epoch: {}'.format(epoch + 1))
    print('*' * 10)
    running_loss = 0
    input_hidden = torch.autograd.Variable(torch.randn(BATCH_SIZE, HIDDEN_UNITS))
    for _,batch in enumerate(train_loader):
        x = torch.autograd.Variable(batch[0])
        y = torch.autograd.Variable(batch[1])
        # forward
        out, input_hidden = rnnmodel(x, input_hidden)
        trgt = torch.max(y, 1)[1]
        loss = criterion(out, trgt)
        running_loss += loss.data[0]
        # backward
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
    print('Loss: {:.6f}'.format(running_loss / len(word_to_ix)))

#print(rnnmodel.state_dict().keys())


f = open("res-0106-rnn-v1.txt","w+")
alpha = rnnmodel.state_dict()['rnn.weight_ih_l0']
for word in sentence_set:
    #print(word,torch.unsqueeze(alpha[word_to_ix[word]],0).numpy())
    line = word + "*" +str(torch.unsqueeze(alpha[word_to_ix[word]],0).numpy().tolist()[0])+"\n"
    #print(line)
    f.write(line)
f.close()
#'''