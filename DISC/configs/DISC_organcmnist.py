algorithm = 'DISC'
# dataset param
dataset = 'organcmnist'
input_channel = 3
num_classes = 11
# root = '/data/yfli/CIFAR10'
noise_type = 'instance'
noise_rate = 0.2
seed = 0
loss_type = 'ce'
# model param
# model1_type = 'resnet18'
# model2_type = 'none'
# train param
gpu = '0'
batch_size = 128
lr = 0.01
# lr = 0.0004
epochs = 50
num_workers = 4
adjust_lr = 1
epoch_decay_start = 30
alpha = 5.0
# result param
save_result = False