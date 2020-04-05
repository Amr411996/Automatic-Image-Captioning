import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)

        # the LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(embed_size, self.hidden_size, num_layers=self.num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        # initialize the weights
        #self.init_weights()
        
        # initialize the hidden state
        self.hidden = (torch.zeros(1, 1, hidden_size),torch.zeros(1, 1, hidden_size)) 
    
    def forward(self, features, captions):
      
        # get the output and hidden state by passing the lstm over our word embeddings
        # the lstm takes in our embeddings and hiddent state
        captions = captions[:, :-1]
        #print(captions.shape,features.shape)
        #features = features.view(features.shape[0], 1, features.shape[1])
        embeds = self.word_embeddings(captions)  # 10, 17, 256
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
       
        #print(inputs.shape)
        lstm_out, self.hidden = self.lstm(inputs)
        #print(lstm_out.shape)
        out = self.fc(lstm_out)
        return out

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        output_list = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)
            #print(lstm_out.shape)
            lstm_out = lstm_out.squeeze(1)
            fc_out = self.fc(lstm_out)
            word_id = fc_out.max(1)[1]
            output_list.append(word_id.item())
            inputs = self.word_embeddings(word_id).unsqueeze(1)
              
        return output_list
    
