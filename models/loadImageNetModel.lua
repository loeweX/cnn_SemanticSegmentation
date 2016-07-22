require 'loadcaffe'
require 'cudnn'
require 'nn'
require 'cunn'
require 'torch'

print '==> Loading network'
--net = loadcaffe.load("ResNet-50-deploy.prototxt", "ResNet-50-model.caffemodel", 'nn')
--net = loadcaffe.load("VGG_ILSVRC_16_layers_deploy.prototxt", "VGG_ILSVRC_16_layers.caffemodel", 'nn')
net = loadcaffe.load("ResNet-50-deploy.prototxt", "ResNet-50-model.caffemodel", 'nn')
net = cudnn.convert(net, cudnn)

-- print net structure
print(net)

torch.save("resnet-50-orig.t7", net)