import cv2
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image
import torch.nn as nn
predicted_class = 'aa'

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(1024)
        self.relu6 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=4 * 4 * 1024, out_features=512)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.maxpool1(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.maxpool2(x)
        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.maxpool3(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu7(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        return x

# Load the saved model
checkpoint = torch.load('my_checkpoint_cosineLR_v2.pth', map_location=torch.device('cpu'))

# Create a new instance of  model
model = Net()

# Load the state dictionary into model
model.load_state_dict(checkpoint['model_state_dict'])

# Set your model to evaluation mode
model.eval()

# Define the transform to be applied to the image
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Define a function to predict the class of an image
def predict_image(image):
    # Apply the transform to the image
    image_tensor = transform(image)
    # Add a batch dimension to the image
    image_tensor = image_tensor.unsqueeze(0)
    # Make a prediction using the model
    with torch.no_grad():
        output = model(image_tensor)
        # print(output)
        probabilities = F.softmax(output, dim=1)
        predicted_index = torch.argmax(probabilities).item()
        probability = probabilities[0][predicted_index].item()
    # Map the index to the corresponding class name
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = class_names[predicted_index]
    # Return the predicted class name
    return predicted_class, probability
# Define the video capture device
cap = cv2.VideoCapture(0)
# Loop to capture images and make predictions
while True:
    # Capture a frame from the camera
    ret, frame = cap.read()
    # Crop the center square of size 224x224
    height, width = frame.shape[:2]
    crop_size = min(height, width)
    y = (height - crop_size) // 2
    x = (width - crop_size) // 2
    crop_img = frame[y:y + crop_size, x:x + crop_size]
    square_img = cv2.resize(crop_img, (320, 320))

    # Wait for a key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Convert the frame to a PIL image
    image = Image.fromarray(cv2.cvtColor(square_img, cv2.COLOR_BGR2RGB))
    # Make a prediction using the model
    predicted_class, probability = predict_image(image)
    # Print the predicted class
    print('Predicted class:', predicted_class, round(probability * 100, 2))
    text = round(probability * 100, 2)
    # Show the resulting square image
    cv2.putText(square_img, predicted_class, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.putText(square_img, str(text), (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Square Image', square_img)

# Release the capture device and close the window
cap.release()
cv2.destroyAllWindows()



