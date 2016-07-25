model = torch.load("models/ImageNet.t7")

--[[
ToDo: 
1) Delete last layer (Softmax)
2) Change fully-connected layers to convolution layers (according to Caffe definition)
3) Append a 1 × 1 convolution with channel dimension 21 to predict scores for each of the PASCAL classes (including background)
4) Append deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs
]]--

-- 1)
model:remove() --no layer specified -> deletes last (softmax layer)

-- 2)
--[[ Definition of the last layers
  (32): nn.View
  (33): nn.Linear(25088 -> 4096)
  (34): cudnn.ReLU
  (35): nn.Dropout(0.500000)
  (36): nn.Linear(4096 -> 4096)
  (37): cudnn.ReLU
  (38): nn.Dropout(0.500000)
  (39): nn.Linear(4096 -> 1000)
  (40): cudnn.SoftMax
]]--
model:remove(32) -- view-layer converts model to be used in Linear layer --> drop

model:remove(32)
model:insert(cudnn.SpatialConvolution(512,4096, 7,7, 1,1, 1,1), 32)

model:remove(35)
model:insert(cudnn.SpatialConvolution(4096,4096, 1,1, 1,1, 1,1), 35)

-- 3)
model:remove()
model:add(cudnn.SpatialConvolution(4096,opt.numClasses, 1,1, 1,1, 1,1))

-- 4)
model:add(nn.SpatialFullConvolution(opt.numClasses, opt.numClasses, 63, 63))
