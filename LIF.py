import spaic
import torch
import numpy as np
run_time = 1000.0
backend_dt = 0.1
class TestNet(spaic.Network):
    def __init__(self):
        super(TestNet, self).__init__()
        # 输入部分选用 null_encoder, 不对输入进行编码，直接输入编码后的数据
        self.in_node = spaic.Encoder(num=1, coding_method='null')
        self.layer1 = spaic.NeuronGroup(1, model='lif')
        self.connection1 = spaic.Connection(
        self.in_node, self.layer1,
        link_type='full', weight=torch.tensor([[1]]))
        # Monitor
        self.mon_V = spaic.StateMonitor(self.layer1, 'V')
        self.mon_in = spaic.StateMonitor(self.in_node, 'O')
        self.mon_L1 = spaic.StateMonitor(self.layer1, 'O')
        # 设置仿真依赖的后端
        self.set_backend(spaic.Torch_Backend('cpu'))
        self.set_backend_dt(dt=backend_dt)

Net = TestNet()
my_input_data = torch.zeros([int(run_time/backend_dt)])
my_input_data[int(10/backend_dt):int(60/backend_dt)] = 0.006
my_input_data[int(110/backend_dt):int(160/backend_dt)] = 0.008
my_input_data[int(210/backend_dt):int(260/backend_dt)] = 0.010
my_input_data[int(310/backend_dt):int(360/backend_dt)] = 0.012
my_input_data[int(410/backend_dt):int(460/backend_dt)] = 0.014
my_input_data[int(510/backend_dt):int(560/backend_dt)] = 0.016
my_input_data[int(660/backend_dt):int(760/backend_dt)] = 0.004
my_input_data[int(760/backend_dt):int(860/backend_dt)] = 0.008
my_input_data[int(860/backend_dt):int(940/backend_dt)] = 0.006
Net.in_node(my_input_data.view(1, 10000, 1))
Net.run(run_time)

time_line = Net.mon_V.times # 监视器监视事件的时间戳
value_line = Net.mon_V.values[0][0] # 两个 [0] 分别代表在批次中第 0 个样本
input_line = Net.mon_in.values[0][0] # 以及在神经元组中第 0 个神经元
output_time = Net.mon_L1.times
output_line = Net.mon_L1.values[0][0]


import matplotlib.pyplot as plt
plt.subplot(3, 1, 1)
plt.title('Leaky Integrated-and-Fire Model')
plt.plot(time_line, input_line, label='input current')
plt.ylabel("Current")
plt.legend()
plt.subplot(3, 1, 2)
plt.plot(time_line, value_line, label='V')
plt.ylabel("Membrane potential")
plt.ylim((-0.1, 1.5))
plt.legend()
plt.subplot(3, 1, 3)
plt.plot(output_time, output_line, label='output spike')
plt.xlabel("time")
plt.legend()
plt.show()