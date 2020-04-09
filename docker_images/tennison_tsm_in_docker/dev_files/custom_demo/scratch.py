import torch
from mobilenet_v2_tsm_test import MobileNetV2

#
#model = torch.load("mobilenetv2_jester_online.pth.tar")

#model = torch.load("../../pretrained/2cat/ckpt.best.pth.tar")['state_dict']

# Trained 9 category model
model_weights = torch.load("../../pretrained/9cat/ckpt.best.pth.tar")['state_dict']

# Checkpoint from website
#model = torch.load("TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth")["state_dict"]

#for k,v in model.items():
#    print(k)


actual_model = MobileNetV2(n_class=10)
#print(model.state_dict())

print(len(model_weights))
print(len(actual_model.state_dict()))

for k1, k2 in zip(model_weights.keys(), actual_model.state_dict().keys()):
    print(k1, k2)



#print(torch.cat((torch.ones(3, 2), torch.zeros(2, 2), torch.ones(4,2)), 0))
#the_ones = torch.ones(3,2)
#the_zeros = torch.zeros(2,2)

