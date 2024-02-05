import spaic
from spaic.IO.Dataset import MNIST as dataset
from spaic.Learning.Learner import Learner
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()
# 设备设置
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
backend = spaic.Torch_Backend(device)
backend.dt = 0.1
run_time = 20.0
bat_size = 200
# 创建训练数据集
root = '/home/qyy/Documents/brain-inspired-network/mnist/MNIST/raw'
train_set = dataset(root, is_train=False)
test_set = dataset(root, is_train=False)
train_loader = spaic.Dataloader(train_set, batch_size=bat_size, shuffle=True, drop_last=False)
test_loader = spaic.Dataloader(test_set, batch_size=bat_size, shuffle=False)

node_num = dataset.maxNum
label_num = dataset.class_number

class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()
        self.input = spaic.Encoder(num=node_num,
        coding_method='poisson')
        self.layer1 = spaic.NeuronGroup(label_num, model='clif')
        self.connection1 = spaic.Connection(self.input, self.layer1, link_type='full',
        w_std=0.08, w_mean=0.02)
        self.output = spaic.Decoder(num=label_num, dec_target=self.layer1,
        coding_method='spike_counts')
        # Learner
        self._learner = Learner(algorithm='Tempotron',
        trainable=[self.connection1, self.output], lr=0.05,tau=60,tau_s=20)
        self.set_backend(backend)

Net = TestNet()
for epoch in range(20):
    pbar = tqdm(total=len(train_loader))
    train_acc = 0
    for i, item in enumerate(train_loader):
        data, label = item
        Net.input(data)
        Net.output(label)
        Net.run(run_time)
        output = Net.output.predict
        label = torch.tensor(label, device=device,
        dtype=torch.long)
        predict_labels = torch.argmax(output, 1)
        num_correct = (predict_labels == label.view(-1)).sum().item()
        acc = num_correct / data.shape[0]
        train_acc += acc
        pbar.set_description_str("[Acc:%.2f]Batch progress: " % (acc*100))
        pbar.update()
    pbar.close()
    Net._learner.lr=Net._learner.lr*0.9
    print(Net._learner.lr)
    Net.save_state('model', '/home/qyy/Documents/brain-inspired-network/runs/temp1/', True)
    print('\n[%d]train_acc: %f' %(epoch, train_acc / len(train_loader)))
    writer.add_scalar("Acc", train_acc / len(train_loader), epoch)
writer.flush()
writer.close()