require 'torch'
require 'nn'
require 'image'
require 'cudnn'
require 'cutorch'
require 'cunn'
--display = require 'display'

--th -ldisplay.start 8000 0.0.0.0
--http://localhost:8000/

cudnn.benchmark = true
cudnn.fastest = true

numClasses = 21
tsImage = torch.ByteTensor(250,250,3)
local modelPath = 'myModel.t7'
local meanPath = 'meanStd.t7'
local mean
local model
local img_size_2
local img_size_3

local confidence = 0.022
model = torch.load(modelPath)
model:add(cudnn.SpatialSoftMax())
model:evaluate()
model:cuda()


print('==> Loading mean') 
local meanstd = torch.load(meanPath)
mean = meanstd.mean



function preprocess(img) 

  img = img:float()
  img = img:transpose(1,3)
  img = img:transpose(2,3)

  img_size_2 = img:size(2)
  img_size_3 = img:size(3)

  -- rescale the image
  img = image.scale(img,224,224)
  
  --image comes from openCV -> already in BGR, but mean in RGB
  img[{1,{},{}}]:add(-(mean[3]*255))
  img[{2,{},{}}]:add(-(mean[2]*255))
  img[{3,{},{}}]:add(-(mean[1]*255))

  return img
end
  
input = torch.CudaTensor(1,3,224,224)

function classify()
  img = preprocess(tsImage)
  input[1] = img

  output = model:forward(input)
  -- _,prediction_sorted = output:float():sort(2, true)

  --------------------------------------------
  -- uses only the confidence values for human label - if higher than confidence, label as human
  tmpOut = output[1]
  out = tmpOut[16]

  mask = out:gt(confidence):byte()
  finalOut = torch.ones(224,224)
  finalOut:maskedFill(mask, 16)  
  --------------------------------------------

  --------------------------------------------
  -- alternatively, take only those pixels as humans, where the confidence for human is biggest of all classes
  --finalOut = prediction_sorted[{1,1,{},{}}]
  --do this only if you want only humans to be segmented!
  --mask = prediction_sorted[{1,1,{},{}}]:ne(16):byte() --16 = label for human
  --finalOut:maskedFill(mask, 1)
  ------------------------------------------------------

  outTmp = torch.Tensor(1,224,224)
  outTmp[1] = finalOut
  finalOut = image.scale(outTmp, img_size_3, img_size_2, 'simple')

  return finalOut[1]:byte()
end

