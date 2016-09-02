-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

--[[
Loading dataset for training

requirements:
- square input images with corresponding label images (where each pixel value corresponds to the label at that position)
- txt file with all image names (train.txt, val.txt)
  one row for each image/label pair, seperated by space

- change paths according to your dataset (dataPath, imagePath, meanStdFile)
- check if all transformations in trainHook/valHool work with your dataset (especially the label transformations - all label values have to be >= 1)
]]--

require 'image'
require 'xlua'

-- provide correct paths
local dataPath = '/data/DNN-common/DeconvPascal2012/imagesets/stage_1_train_imgset' --Stage 1 Training
local imagePath = '/data/DNN-common/DeconvPascal2012/VOC2012'

-- if not provided, will create meanStd file on the first run
local meanStdFile = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation/meanStd.t7'

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainImagesFile = paths.concat(dataPath, 'trainImages.t7')
local valImagesFile = paths.concat(dataPath, 'valImages.t7')

trainImages = {}
trainLabels = {}

if paths.filep(trainImagesFile) then
  print('Loading trainImage metadata from cache')
  trainLoader = torch.load(trainImagesFile)
  trainImages = trainLoader.trainImages
  trainLabels = trainLoader.trainLabels
else
  print('Creating trainImage metadata')
  trainFile = assert(io.open(paths.concat(dataPath, "train.txt")))
  counter = 1
  trainImages = {}
  trainLabels = {}

  for line in trainFile:lines() do
    trainImages[counter], trainLabels[counter] = unpack(line:split(" "))
    counter = counter + 1
  end
  trainFile:close()

  local trainLoader = {}
  trainLoader.trainImages = trainImages
  trainLoader.trainLabels = trainLabels
  torch.save(trainImagesFile, trainLoader)
end


if paths.filep(valImagesFile) then
  print('Loading valImage metadata from cache')
  valLoader = torch.load(valImagesFile)
  valImages = valLoader.valImages
  valLabels = valLoader.valLabels
else
  print('Creating valImage metadata')
  valFile = assert(io.open(paths.concat(dataPath, "val.txt")))
  counter = 1
  valImages = {}
  valLabels = {}

  for line in valFile:lines() do
    valImages[counter], valLabels[counter] = unpack(line:split(" "))
    counter = counter + 1
  end
  valFile:close()

  local valLoader = {}
  valLoader.valImages = valImages
  valLoader.valLabels = valLabels
  torch.save(valImagesFile, valLoader)
end



if paths.filep(meanStdFile) then
  local meanstd = torch.load(meanStdFile)
  mean = meanstd.mean
  std = meanstd.std
  print('Loaded mean and std from cache')
else
  local tm = torch.Timer()  
  splitSize = 10000  

  print('Estimating the mean (per-channel, shared for all pixels) over ' .. splitSize .. ' randomly sampled training images')

  trainSplit = torch.Tensor(splitSize,3,250,250)
  for i = 1,splitSize do  --estimate mean and std on a subset of the dataset
    tmp = image.load(paths.concat(imagePath, string.sub(trainImages[i],2,-1)))
    tmp = image.scale(tmp, 250,250,'simple')
    trainSplit[i] = tmp
    xlua.progress(i, splitSize)
  end

  channels = {'r','g','b'}
  image_mean = {}
  image_std = {}
  for i,channel in ipairs(channels) do
    -- normalize each channel globally:
    image_mean[i] = trainSplit[{ {},i,{},{} }]:mean()
    image_std[i] = trainSplit[{ {},i,{},{} }]:std()
  end

  local cache = {}
  cache.mean = image_mean
  cache.std = image_std
  torch.save(meanStdFile, cache)
  print('Time to estimate:', tm:time().real)
end

collectgarbage()

-----------------------------------------------------------------------------------------------------------------

-- function to load single train image, jitter it appropriately (random crops etc.)
local function trainHook(index)
  collectgarbage()
  local tmpInput = image.load(paths.concat(imagePath, string.sub(trainImages[index],2,-1)))
  local tmpLabel = image.load(paths.concat(imagePath, string.sub(trainLabels[index],2,-1)))

  assert(tmpInput:size(3) == tmpLabel:size(3))
  assert(tmpInput:size(2) == tmpLabel:size(2))

  -- labels must be integer values >= 1
  tmpLabel = tmpLabel * 255
  tmpLabel = tmpLabel + 1
  -- opt.numClasses + 1 label will not be learned and does not contribute to error calculation (void label)
  tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1

  -- scale input to be slightly bigger than wanted and then do a random crop
  tmpInput = image.scale(tmpInput,opt.targetSize+25,opt.targetSize+25)
  tmpLabel = image.scale(tmpLabel,opt.targetSize+25,opt.targetSize+25,'simple')

  local iW = tmpInput:size(3)
  local iH = tmpInput:size(2)

  -- do random crop
  local oW = opt.targetSize
  local oH = opt.targetSize
  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
  local outImg = image.crop(tmpInput, w1, h1, w1 + oW, h1 + oH)
  local outLab = image.crop(tmpLabel, w1, h1, w1 + oW, h1 + oH)

  assert(outImg:size(3) == oW)
  assert(outImg:size(2) == oH)

  -- do horizontal flip with probability 0.5
  if torch.uniform() > 0.5 then 
    outImg = image.hflip(outImg) 
    outLab = image.hflip(outLab) 
  end
  
  -- subtract mean
  for i=1,3 do -- channels
    if mean then outImg[{{i},{},{}}]:add(-mean[i]) 
    else error('no mean given')
    end
  end

  outImg = outImg:index(1,torch.LongTensor{3,2,1}) --rgb to bgr
  outImg = outImg * 255  -- [0,1] to [0,255]

  return outImg, outLab
end

-- function that is called from the train function, returns batch of training images+labels
function loadTrainBatch(indices)
  inputs = torch.Tensor(opt.batchSize,3,opt.targetSize,opt.targetSize)
  labels = torch.Tensor(opt.batchSize,opt.targetSize,opt.targetSize)

  for i = 1,opt.batchSize do
    inputs[i], labels[i] = trainHook(indices[i])
  end
  return inputs, labels
end


-- function to load val image (similar to trainHook, but without random crop and horizontal flip)
local function valHook(index)
  collectgarbage()
  local tmpInput = image.load(paths.concat(imagePath, string.sub(valImages[index],2,-1)))
  local tmpLabel = image.load(paths.concat(imagePath, string.sub(valLabels[index],2,-1)))

  assert(tmpInput:size(3) == tmpLabel:size(3))
  assert(tmpInput:size(2) == tmpLabel:size(2))

  tmpLabel = tmpLabel * 255
  tmpLabel = tmpLabel + 1
  tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1

  tmpInput = image.scale(tmpInput,opt.targetSize,opt.targetSize)
  tmpLabel = image.scale(tmpLabel,opt.targetSize,opt.targetSize,'simple')

  -- mean/std
  for i=1,3 do -- channels
    if mean then tmpInput[{{i},{},{}}]:add(-mean[i]) 
    else error('no mean given')
    end
  end
  
  tmpInput = tmpInput:index(1,torch.LongTensor{3,2,1}) --rgb to bgr
  tmpInput = tmpInput * 255
  
  return tmpInput, tmpLabel, valImages[index]
end

function loadValBatch(indices)  
  inputs = torch.Tensor(opt.batchSizeVal,3,opt.targetSize,opt.targetSize)
  labels = torch.Tensor(opt.batchSizeVal,opt.targetSize,opt.targetSize)
  imNames = {}

  for i = 1,opt.batchSizeVal do
    inputs[i], labels[i], imNames[i] = valHook(indices[i])
  end  

  return inputs, labels, imNames
end
