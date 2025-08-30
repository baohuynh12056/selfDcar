from ai.model import DrivingModel
import torch

def train():
    model = DrivingModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(10):
        # Giả lập batch dữ liệu
        inputs = torch.rand((16, 10))  # 16 sample, 10 features
        labels = torch.randint(0, 3, (16,))  # 3 classes

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "assets/weights/driving_model.pt")
