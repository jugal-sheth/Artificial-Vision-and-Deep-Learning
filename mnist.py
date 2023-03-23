import cv2
import torch
import torchvision.transforms as transforms
import torch.nn as nn


# neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.dropout = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = nn.functional.relu(nn.functional.max_pool2d(self.conv1(x), 2))
        x = nn.functional.relu(nn.functional.max_pool2d(self.dropout(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = nn.functional.relu(self.fc1(x))
        x = nn.functional.dropout(x, training=self.training)
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)


# Load model
model = Net()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()


def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    resized = cv2.resize(threshold, (28, 28), interpolation=cv2.INTER_AREA)
    cv2.imshow("resized", resized)
    tensor = transforms.ToTensor()(resized).unsqueeze(0)
    tensor = transforms.Normalize((0.5,), (0.5,))(tensor)
    return tensor


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    tensor = preprocess_image(frame)
    with torch.no_grad():
        output = model(tensor)
        prediction = output.argmax(dim=1, keepdim=True).item()
    cv2.putText(frame, str(prediction), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 4)
    cv2.imshow('input', frame)

    # exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
