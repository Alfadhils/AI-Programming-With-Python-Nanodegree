import argparse
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from collections import OrderedDict
from torch.utils.data import DataLoader
import os

def load_data(data_dir):
    # Load datasets
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Configure image augmentations
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Configure image folder object
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)

    # Create dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=32)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

    return trainloader, validloader, testloader, train_data.class_to_idx
    
def build_model(arch, hidden_units):
    if arch == 'VGG':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
            
        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier
        
        return model

    elif arch == 'Densenet':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model.classifier[0].in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.classifier = classifier
        
        return model

    elif arch == 'ResNet':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(model.fc.in_features, hidden_units)),
            ('relu', nn.ReLU()),
            ('dropout', nn.Dropout(0.2)),
            ('fc2', nn.Linear(hidden_units, 102)),
            ('output', nn.LogSoftmax(dim=1))
        ]))

        model.fc = classifier
        
        return model

        
def validate_model(model, criterion, loader, device):
    # Calculate test loss
    valid_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)

            valid_loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(f"Validation loss: {valid_loss/len(loader):.3f}.. "
          f"Validation accuracy: {accuracy/len(loader):.3f}")
    
def train_model(model, criterion, optimizer, loader, epochs, device):
    model.to(device)
    
    epochs = epochs
    steps = 0
    train_losses, valid_losses = [], []
    running_loss = 0
    print_every = 20
    
    try:
        # Train loop
        for epoch in range(epochs):
            for inputs, labels in loader['train']:
                steps += 1
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                logps = model(inputs)
                loss = criterion(logps, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

                # Validation loop
                if steps % print_every == 0:
                    valid_loss = 0
                    accuracy = 0
                    model.eval()
                    with torch.no_grad():
                        for inputs, labels in loader['valid']:
                            inputs, labels = inputs.to(device), labels.to(device)
                            logps = model(inputs)
                            batch_loss = criterion(logps, labels)

                            valid_loss += batch_loss.item()

                            # Calculate accuracy
                            ps = torch.exp(logps)
                            top_p, top_class = ps.topk(1, dim=1)
                            equals = top_class == labels.view(*top_class.shape)
                            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                    train_losses.append(running_loss/print_every)
                    valid_losses.append(valid_loss/len(loader['valid']))
                    print(f"Epoch {epoch+1}/{epochs}.. "
                          f"Train loss: {running_loss/print_every:.3f}.. "
                          f"Validation loss: {valid_loss/len(loader['valid']):.3f}.. "
                          f"Validation accuracy: {accuracy/len(loader['valid']):.3f}")

                    running_loss = 0
                    model.train()
                    
        print('Training complete, now validating..')
        validate_model(model, criterion, loader['valid'], device)
        
    except Exception as e:
        print(f"An error occurred during training: {e}")
        
        
def save_model(model, path, optimizer, arch):
    checkpoint = {
        'arch' : arch,
        'classifier' : model.classifier,
        'state_dict' : model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        'class_to_idx' : model.class_to_idx
    }
    
    # Save trained model
    torch.save(checkpoint, path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument("data_directory", type=str, help="Path to the dataset directory", default='./flowers/')
    parser.add_argument("--save_dir", type=str, help="Directory to save checkpoints", default='./checkpoint_script.pth')
    parser.add_argument("--arch", type=str, help="Choose architecture (default: vgg16)", default="VGG")
    parser.add_argument("--learning_rate", type=float, help="Set learning rate (default: 0.01)", default=0.01)
    parser.add_argument("--hidden_units", type=int, help="Set number of hidden units (default: 512)", default=512)
    parser.add_argument("--epochs", type=int, help="Set number of epochs (default: 20)", default=20)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for training")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    
    train_loader, valid_loader, test_loader, class_to_idx = load_data(args.data_directory)
    model = build_model(args.arch, args.hidden_units)
    model.class_to_idx = class_to_idx
    
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    print(f"Training ({device}) with the following parameters:\n"
      f"  - Architecture: {args.arch}\n"
      f"  - Epochs: {args.epochs}\n"
      f"  - Hidden Units: {args.hidden_units}\n"
      f"  - Learning Rate: {args.learning_rate}")

    train_model(model, criterion, optimizer, {'train': train_loader, 'valid': valid_loader}, args.epochs, device)
    
    print(f"Saving trained model on {args.save_dir}")
    save_model(model, args.save_dir, optimizer, args.arch)