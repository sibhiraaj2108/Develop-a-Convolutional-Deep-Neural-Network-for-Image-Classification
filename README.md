# Develop a Convolutional Deep Neural Network for Image Classification

## AIM
To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images.

##   PROBLEM STATEMENT AND DATASET
Include the Problem Statement and Dataset. Image classification is a fundamental task in computer vision where an input image is assigned to one of several predefined classes. The objective of this experiment is to build and train a Convolutional Neural Network (CNN) using a labeled image dataset and evaluate its performance using accuracy, confusion matrix, and classification report.

## Neural Network Model
<img width="907" height="639" alt="image" src="https://github.com/user-attachments/assets/dbbd6362-5c54-4933-9f11-5b785a0b38df" />


## DESIGN STEPS
### STEP 1: 
Load and Preprocess Data

### STEP 2: 

Get the shape of the first image in the training dataset


### STEP 3: 

Get the shape of the first image in the test dataset


### STEP 4: 

Train the Model


### STEP 5: 

Test the Model


### STEP 6: 

Predict on a Single Image and Display the image.

## PROGRAM

### Name: SIBHIRAAJ R

### Register Number: 212224230268

```
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1)
        self.conv3 = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.fc1 = nn.Linear(128*3*3,128)
        self.fc2 = nn.Linear(128,64)
        self.fc3 = nn.Linear(64,10)
    def forward(self, x):
      x = self.pool(torch.relu(self.conv1(x)))
      x = self.pool(torch.relu(self.conv2(x)))
      x = self.pool(torch.relu(self.conv3(x)))
      x=x.view(x.size(0),-1)
      x=torch.relu(self.fc1(x))
      x=torch.relu(self.fc2(x))
      x=self.fc3(x)
      return x



# Initialize the Model, Loss Function, and Optimizer
model = CNNClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the Model
def train_model(model, train_loader, num_epochs=3):

# write your code here
def train_model(model, train_loader, num_epochs=3):
  for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
      optimizer.zero_grad()
      outputs = model(images)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
    print('Name:SIBHIRAAJ R ')
    print('Register Number:212224230268')
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

```

### OUTPUT

## Training Loss per Epoch

<img width="400" height="195" alt="image" src="https://github.com/user-attachments/assets/4d996f97-e475-4242-b76a-c3c1797961ca" />


## Confusion Matrix

<img width="670" height="630" alt="image" src="https://github.com/user-attachments/assets/f60660c9-3a64-44f7-89ad-0513639eea71" />


## Classification Report

<img width="710" height="379" alt="image" src="https://github.com/user-attachments/assets/6e1dcda1-2b1e-49cc-b5b1-1c6555197e1f" />


### New Sample Data Prediction
<img width="665" height="681" alt="image" src="https://github.com/user-attachments/assets/c767ef69-e5bc-4614-bf78-df6a74d5e93a" />



## RESULT
Thus, To develop a convolutional deep neural network (CNN) for image classification and to verify the response for new images is executed and verified successfully.
