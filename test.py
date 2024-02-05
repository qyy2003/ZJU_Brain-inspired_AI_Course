import spaic
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()

        # Encoding
        self.input = spaic.Encoder(num=784, coding_method='poisson')

        # NeuronGroup
        self.layer1 = spaic.NeuronGroup(num=400, model='lif')
        self.layer2 = spaic.NeuronGroup(num=80, model='lif')
        self.layer3 = spaic.NeuronGroup(num=10, model='clif')

        # Decoding
        self.output = spaic.Decoder(num=10, dec_target=self.layer3, coding_method='spike_counts')

        # Connection
        self.connection1 = spaic.Connection(pre=self.input, post=self.layer1, link_type='full',
        w_std=0.05, w_mean=0.02)
        self.connection2 = spaic.Connection(pre=self.layer1, post=self.layer2, link_type='full',
        w_std=0.05, w_mean=0.02)
        self.connection3 = spaic.Connection(pre=self.layer2, post=self.layer3, link_type='full',
        w_std=0.05, w_mean=0.02)

        # Monitor
        # self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        # self.spk_O = spaic.SpikeMonitor(self.layer1, 'O')

        # Learner
        self.learner = spaic.Learner(trainable=self, algorithm='STCA')
        self.learner.set_optimizer('Adam', 0.002)
        self.learner.set_schedule('StepLR', step_size=600, gamma=0.6)#0.5&0.6&0.7
        if torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        backend = spaic.Torch_Backend(device)
        backend.dt = 0.1

        self.set_backend(backend)
if __name__=="__main__":
    # Network instantiation
    Net = TestNet()
    from tqdm import tqdm
    import torch.nn.functional as F
    from spaic.IO.Dataset import MNIST as dataset
    # Create the training data set
    root = '/home/qyy/Documents/brain-inspired-network/mnist/MNIST/raw'
    import os
    print(os.path.join(root))
    print(os.path.exists(os.path.join(root)))
    train_set = dataset(root, is_train=True)
    test_set = dataset(root, is_train=False)

    # Set the run time and batch size
    run_time = 50
    bat_size = 100

    # Create the DataLoader iterator
    train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=False)
    test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)
    eval_losses = []
    eval_acces = []
    losses = []
    acces = []
    num_correct = 0
    num_sample = 0
    for epoch in range(10):

        # Train
        print("Start training")
        train_loss = 0
        train_acc = 0
        pbar = tqdm(total=len(train_loader))
        for i, item in enumerate(train_loader):
            # forward propagation
            data, label = item
            Net.input(data)
            Net.output(label)
            Net.run(run_time)
            output = Net.output.predict
            output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
            label = torch.tensor(label, device=device)
            batch_loss = F.cross_entropy(output, label)

            # Back propagation
            Net.learner.optim_zero_grad()
            batch_loss.backward(retain_graph=False)
            Net.learner.optim_step()

            # Record the error
            train_loss += batch_loss.item()
            predict_labels = torch.argmax(output, 1)
            num_correct = (predict_labels == label).sum().item()  # Record the number of correct tags
            acc = num_correct / data.shape[0]
            train_acc += acc

            pbar.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
            pbar.update()
        pbar.close()
        Net.save_state('model', '/home/qyy/Documents/brain-inspired-network/runs/test1/', True)
        losses.append(train_loss / len(train_loader))
        acces.append(train_acc / len(train_loader))
        print('epoch:{},Train Loss:{:.4f},Train Acc:{:.4f}'.format(epoch, train_loss / len(train_loader), train_acc / len(train_loader)))
        writer.add_scalar("Loss/train", train_loss / len(train_loader), epoch)
        writer.add_scalar("Acc/train", train_acc / len(train_loader), epoch)

        # Test
        eval_loss = 0
        eval_acc = 0
        print("Start testing")
        pbarTest = tqdm(total=len(test_loader))
        with torch.no_grad():
            for i, item in enumerate(test_loader):
                data, label = item
                Net.input(data)
                Net.run(run_time)
                output = Net.output.predict
                output = (output - torch.mean(output).detach()) / (torch.std(output).detach() + 0.1)
                label = torch.tensor(label, device=device)
                batch_loss = F.cross_entropy(output, label)
                eval_loss += batch_loss.item()

                _, pred = output.max(1)
                num_correct = (pred == label).sum().item()
                acc = num_correct / data.shape[0]
                eval_acc += acc
                pbarTest.set_description_str("[loss:%f]Batch progress: " % batch_loss.item())
                pbarTest.update()
            eval_losses.append(eval_loss / len(test_loader))
            eval_acces.append(eval_acc / len(test_loader))
        pbarTest.close()
        print('epoch:{},Test Loss:{:.4f},Test Acc:{:.4f}'.format(epoch,eval_loss / len(test_loader), eval_acc / len(test_loader)))
        writer.add_scalar("Loss/validation", eval_loss / len(test_loader), epoch)
        writer.add_scalar("Acc/validation",   eval_acc / len(test_loader), epoch)
    writer.flush()
    writer.close()
    from matplotlib import pyplot as plt
    plt.subplot(2, 1, 1)
    plt.plot(acces)
    plt.title('Train Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('epoch')

    plt.subplot(2, 1, 2)
    plt.plot(eval_acces)
    plt.title('Test Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('epoch')

    plt.show()