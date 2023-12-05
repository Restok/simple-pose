from preprocess_data import get_datasets
import torch
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2

train_data, test_data = get_datasets()
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
sample_frame, sample_label, _ = next(iter(train_loader))
#print number of test dataset

print('number of train images:', len(train_data))
print('number of test images:', len(test_data))

input_size = sample_frame.shape
output_size = sample_label.shape

print('input size:', input_size) # (batch_size, channels, height, width)
print('output size:', output_size) # (batch_size, 75)

# Create a model
# move to cuda
model = mobilenet_v2(pretrained=True)
model.classifier[1] = torch.nn.Linear(1280, output_size[1])
model = model.to('cuda')
model.train()

# Train the model
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()


#start timing
import time

start = time.time()
last_time = start
for epoch in range(2):
    for i, v in enumerate(train_loader):
        frames, labels, _ = v
        frames = frames.to('cuda')
        labels = labels.to('cuda')
        optimizer.zero_grad()
        outputs = model(frames)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        if i % 50 == 0:
            print('Epoch: %d, Iteration: %d/%d, Loss: %f' % (epoch, i,len(train_loader), loss.item()))
        if i % 100 == 0:
            time_elapsed = time.time() - last_time
            print('Time elapsed: %d seconds' % time_elapsed)
            last_time = time.time()
            #estimate time remaining
            time_remaining = time_elapsed * (len(train_loader) - i)/100
            print('Estimated time remaining for epoch: %d seconds' % time_remaining)
    #get MSE loss on test set
    with torch.no_grad():
        total_loss = 0
        for i, v in enumerate(test_loader):
            frames, labels,_ = v
            frames = frames.to('cuda')
            labels = labels.to('cuda')
            outputs = model(frames)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
        print('Epoch: %d, Test Loss: %f' % (epoch, total_loss / len(test_loader)))

# Save the model checkpoint
torch.save(model.state_dict(), 'checkpoints/mobilenet_v2_224x224_smplx.pth')

# Inference with the saved model
# load model mobilenet_v2_224x224.pth