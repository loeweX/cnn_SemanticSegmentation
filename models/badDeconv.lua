require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'



model = torch.load("models/ImageNet.t7")

--[[ToDo:
- add batchNormalization to the output of every convolutional/deconvolutional layer
- remove dropout
- pool -> unpool
- deconv net
]]--


--BatchNormalization after Linear Layer, last layer is Convolution not FullConvolution!!


-- add BatchNormalization to existing network
model:insert(nn.SpatialBatchNormalization(64), 2)
model:insert(nn.SpatialBatchNormalization(64), 5)
model:insert(nn.SpatialBatchNormalization(128), 9)
model:insert(nn.SpatialBatchNormalization(128), 12)
model:insert(nn.SpatialBatchNormalization(256), 16)
model:insert(nn.SpatialBatchNormalization(256), 19)
model:insert(nn.SpatialBatchNormalization(256), 22)
model:insert(nn.SpatialBatchNormalization(512), 26)
model:insert(nn.SpatialBatchNormalization(512), 29)
model:insert(nn.SpatialBatchNormalization(512), 32)
model:insert(nn.SpatialBatchNormalization(512), 36)
model:insert(nn.SpatialBatchNormalization(512), 39)
model:insert(nn.SpatialBatchNormalization(512), 42) 

-- remove dropout
model:remove(48)
model:remove(50)

model:insert(nn.BatchNormalization(4096), 47) 
model:insert(nn.BatchNormalization(4096), 50) 


--pooling - unpooling
poolLayer = {7,14,24,34,44}
model:remove(poolLayer[1])
pool1 = nn.SpatialMaxPooling(2,2,2,2)
model:insert(pool1,poolLayer[1])

model:remove(poolLayer[2])
pool2 = nn.SpatialMaxPooling(2,2,2,2)
model:insert(pool2,poolLayer[2])

model:remove(poolLayer[3])
pool3 = nn.SpatialMaxPooling(2,2,2,2)
model:insert(pool3,poolLayer[3])

model:remove(poolLayer[4])
pool4 = nn.SpatialMaxPooling(2,2,2,2)
model:insert(pool4,poolLayer[4])

model:remove(poolLayer[5])
pool5 = nn.SpatialMaxPooling(2,2,2,2)
model:insert(pool5,poolLayer[5])


--deconv net
model:remove()
model:remove()

model:add(nn.Reshape(4096,1,1,true))

model:add(cudnn.SpatialFullConvolution(4096, 512, 7, 7)) --deconv-fc6
model:add(nn.SpatialBatchNormalization(512))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool5))

for i = 1,3 do --deconv 5
  model:add(cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(512))
  model:add(cudnn.ReLU(true))
end

model:add(nn.SpatialMaxUnpooling(pool4)) --unpool4


for i = 1,2 do --deconv 4
  model:add(cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(512))
  model:add(cudnn.ReLU(true))
end

model:add(cudnn.SpatialFullConvolution(512, 256, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(256))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool3)) --unpool3


for i = 1,2 do --deconv 3
  model:add(cudnn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(256))
  model:add(cudnn.ReLU(true))
end

model:add(cudnn.SpatialFullConvolution(256, 128, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool2)) --unpool2


--deconv 2
model:add(cudnn.SpatialFullConvolution(128, 128, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(128))
model:add(cudnn.ReLU(true))

model:add(cudnn.SpatialFullConvolution(128, 64, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(64))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool1)) --unpool1


for i = 1,2 do --deconv 1
  model:add(cudnn.SpatialFullConvolution(64, 64, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(64))
  model:add(cudnn.ReLU(true))
end

model:add(cudnn.SpatialConvolution(64, 21, 1, 1))  --kein padding!! 

print(color.blue 'remember last layer Convolution <-> Deconvolution')

--[[ Caffe definition of the last layers 

layers { bottom: 'pool5' top: 'fc6' name: 'fc6' type: CONVOLUTION
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { engine: CAFFE kernel_size: 7 num_output: 4096 } }
layers { bottom: 'fc6' top: 'fc6' name: 'relu6' type: RELU }
layers { bottom: 'fc6' top: 'fc6' name: 'drop6' type: DROPOUT
  dropout_param { dropout_ratio: 0.5 } }
layers { bottom: 'fc6' top: 'fc7' name: 'fc7' type: CONVOLUTION
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { engine: CAFFE kernel_size: 1 num_output: 4096 } }
layers { bottom: 'fc7' top: 'fc7' name: 'relu7' type: RELU }
layers { bottom: 'fc7' top: 'fc7' name: 'drop7' type: DROPOUT
  dropout_param { dropout_ratio: 0.5 } }
layers { name: 'score-fr' type: CONVOLUTION bottom: 'fc7' top: 'score'
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { engine: CAFFE num_output: 21 kernel_size: 1 } }

layers { type: DECONVOLUTION name: 'upsample' bottom: 'score' top: 'bigscore'
  blobs_lr: 0 blobs_lr: 0
  convolution_param { num_output: 21 kernel_size: 64 stride: 32 } }
layers { type: CROP name: 'crop' bottom: 'bigscore' bottom: 'data' top: 'upscore' }
]]--