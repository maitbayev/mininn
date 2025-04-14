# mininn

mininn is a lightweight neural network library built entirely from scratch, with only dependency on NumPy.
Its primary aim is for educational purposes.

An example usage, [see notebook](notebooks/mnist.ipynb) for full code:

```python
import mininn

def train(model, loss, train_loader, optimizer, epoch):
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        inputs = inputs.numpy(force=True)
        targets = targets.numpy(force=True)
        loss_output = loss.forward(model(inputs), targets)
        model.backward(loss.backward())
        optimizer.step()
        if i % 10000 == 0:
            print(f"Epoch {epoch} loss {np.array(loss_output):.5f}")

model = mininn.Sequential([
    mininn.Linear(28 * 28, 200),
    mininn.ReLU(),
    mininn.Linear(200, 20)
])
loss = mininn.CrossEntropyLoss()
optimizer = mininn.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    train(model, loss, train_loader, optimizer, epoch)
```