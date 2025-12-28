import torch.nn as nn

class BidirectionalLSTM(nn.Module):
    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()
        
        # 1. The LSTM Layer
        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        
        # 2. The Linear Layer (embedding)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        # The LSTM returns a tuple: (data, hidden_state)
        # We only need the data (recurrent)
        recurrent, _ = self.rnn(input)
        
        # Reshape for the Linear layer
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)
        
        # Pass through Linear layer
        output = self.embedding(t_rec)  # [T * b, nOut]
        
        # Reshape back to [Time, Batch, Features]
        output = output.view(T, b, -1)
        
        return output

class CRNN(nn.Module):
    def __init__(self, img_channel, img_height, img_width, num_class):
        super(CRNN, self).__init__()
        
        # --- CNN (Extracts Visual Features) ---
        self.cnn = nn.Sequential(
            nn.Conv2d(img_channel, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)), 
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)), 
            
            nn.Conv2d(512, 512, kernel_size=2, padding=0), 
            nn.ReLU()
        )

        # --- RNN (Reads the Text) ---
        # We use our custom BidirectionalLSTM wrapper here
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, 256, 256),
            BidirectionalLSTM(256, 256, num_class)
        )

    def forward(self, x):
        # 1. Apply CNN
        conv = self.cnn(x)
        
        # 2. Reshape for RNN
        # [Batch, Channel, Height, Width] -> [Width, Batch, Channel]
        conv = conv.squeeze(2) 
        conv = conv.permute(2, 0, 1) 
        
        # 3. Apply RNN
        # IMPORTANT: We do not unpack with comma (output, _) because 
        # our custom BidirectionalLSTM returns just the tensor.
        output = self.rnn(conv)
        
        return output