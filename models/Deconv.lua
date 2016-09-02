require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'

-- from: http://cvlab.postech.ac.kr/research/deconvnet/model/VGG_conv/VGG_ILSVRC_16_layers_conv.caffemodel
-- and transformed using loadImageNetModel.lua
model = torch.load("models/ImageNetConv.t7") 

model:insert(nn.SpatialBatchNormalization(4096),33)
model:insert(nn.SpatialBatchNormalization(4096),37)


--remove last classification layers
model:remove(35)
model:remove()
model:remove()
model:remove()


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


model:add(cudnn.SpatialFullConvolution(4096, 512, 7, 7)) --deconv-fc6
model:add(nn.SpatialBatchNormalization(512))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool5))

for i = 1,3 do --deconv 5
  model:add(cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(512))
  model:add(nn.ReLU(true))
end

model:add(nn.SpatialMaxUnpooling(pool4)) --unpool4


for i = 1,2 do --deconv 4
  model:add(cudnn.SpatialFullConvolution(512, 512, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(512))
  model:add(nn.ReLU(true))
end

model:add(cudnn.SpatialFullConvolution(512, 256, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(256))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool3)) --unpool3


for i = 1,2 do --deconv 3
  model:add(cudnn.SpatialFullConvolution(256, 256, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(256))
  model:add(nn.ReLU(true))
end

model:add(cudnn.SpatialFullConvolution(256, 128, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool2)) --unpool2


--deconv 2
model:add(cudnn.SpatialFullConvolution(128, 128, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(128))
model:add(nn.ReLU(true))

model:add(cudnn.SpatialFullConvolution(128, 64, 3, 3, 1, 1, 1, 1))  
model:add(nn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))

model:add(nn.SpatialMaxUnpooling(pool1)) --unpool1


for i = 1,2 do --deconv 1
  model:add(cudnn.SpatialFullConvolution(64, 64, 3, 3, 1, 1, 1, 1))  
  model:add(nn.SpatialBatchNormalization(64))
  model:add(nn.ReLU(true))
end

model:add(cudnn.SpatialConvolution(64, 21, 1, 1))  --kein padding!! 

local function ConvInit(name)
  for k,v in pairs(model:findModules(name)) do
   -- local n = v.kW*v.kH*v.nOutputPlane
    v.weight:normal(0,0.01)
    v.bias:zero()
  end
end
local function BNInit(name)
  for k,v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:fill(0.001)
   -- v.momentum = 1
  end
end

ConvInit('cudnn.SpatialFullConvolution')
BNInit('nn.SpatialBatchNormalization')

