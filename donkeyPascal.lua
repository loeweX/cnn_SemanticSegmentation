require 'image'
require 'xlua'

dataPath = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation'
imagePath = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/myImages'
labelPath = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/mySegmentationClass'

-- a cache file of the training metadata (if doesnt exist, will be created)
local trainImagesFile = paths.concat(dataPath, 'trainImages.t7')  --trainVAL...
local valImagesFile = paths.concat(dataPath, 'valImages.t7')
local meanStdFile = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/batchSizeVal/meanStd.t7' -- opt.meanFilePath

if paths.filep(trainImagesFile) then
  print('Loading trainImage metadata from cache')
  trainLoader = torch.load(trainImagesFile)
  trainImages = trainLoader.trainImages
else
  print('Creating trainImage metadata')
--Load Train
  trainFile = assert(io.open(paths.concat(dataPath, "train.txt")))
  counter = 1
  trainImages = {}

  for line in trainFile:lines() do
    trainImages[counter] = line
    -- if counter == 5 then break end  --to load only small subset of images
    counter = counter + 1
  end
  trainFile:close()

  local trainLoader = {}
  trainLoader.trainImages = trainImages
  torch.save(trainImagesFile, trainLoader)
end

if paths.filep(valImagesFile) then
  print('Loading valImage metadata from cache')
  valLoader = torch.load(valImagesFile)
  valImages = valLoader.valImages
else
  print('Creating valImage metadata')
--Load val
  valFile = assert(io.open(paths.concat(dataPath, "val.txt")))
  counter = 1
  valImages = {}

  for line in valFile:lines() do
    valImages[counter] = line
    -- if counter == 5 then break end  --to load only small subset of images
    counter = counter + 1
  end
  valFile:close()

  local valLoader = {}
  valLoader.valImages = valImages
  torch.save(valImagesFile, valLoader)
end



if paths.filep(meanStdFile) then
  local meanstd = torch.load(meanStdFile)
  mean = meanstd.mean
  std = meanstd.std
  print('Loaded mean and std from cache')
else
  local tm = torch.Timer()  
  splitSize = #trainImages 

  print('Estimating the mean (per-channel, shared for all pixels) over ' .. splitSize .. ' randomly sampled training images')

  trainSplit = torch.Tensor(splitSize,3,250,250)
  for i = 1,splitSize do  --estimate mean and std on a subset of the dataset
    tmp = image.load(imagePath .. '/' .. trainImages[i] .. '.jpg')
    tmp = image.scale(tmp,250,250,'simple')
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

-- function to load train image, jitter it appropriately (random crops etc.)
local function trainHook(index)
  collectgarbage()
  local tmpInput = image.load(imagePath .. '/' .. trainImages[index] .. '.jpg')
  local tmpLabel = image.load(labelPath .. '/' .. trainImages[index] .. '.png')

  assert(tmpInput:size(3) == tmpLabel:size(3))
  assert(tmpInput:size(2) == tmpLabel:size(2))

  tmpLabel = tmpLabel * 255
  tmpLabel = tmpLabel + 1
  tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1

  local outImg = tmpInput
  local outLab = tmpLabel
  
  if opt.targetSize < tmpInput:size(2) then
    tmpInput = image.scale(tmpInput,250,250) --different size to have more space for random crops??
    tmpLabel = image.scale(tmpLabel,250,250,'simple')

    local iW = tmpInput:size(3)
    local iH = tmpInput:size(2)

    -- do random crop
    local oW = opt.targetSize
    local oH = opt.targetSize
    local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
    local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
    outImg = image.crop(tmpInput, w1, h1, w1 + oW, h1 + oH)
    outLab = image.crop(tmpLabel, w1, h1, w1 + oW, h1 + oH)
  
    assert(outImg:size(3) == oW)
    assert(outImg:size(2) == oH)
    assert(outLab:size(3) == oW)
    assert(outLab:size(2) == oH)
  end  

-- do hflip with probability 0.5
if torch.uniform() > 0.5 then 
  outImg = image.hflip(outImg) 
  outLab = image.hflip(outLab) 
end
-- mean/std
for i=1,3 do -- channels
  if mean then outImg[{{i},{},{}}]:add(-mean[i]) 
  else error('no mean given')
  end
  --  if std then outImg[{{i},{},{}}]:div(std[i]) 
  --  else error('no std given')
  --  end
end
outImg = outImg:index(1,torch.LongTensor{3,2,1}) --rgb to bgr
outImg = outImg * 255
return outImg, outLab
end

function loadTrainBatch(indices)
  inputs = torch.Tensor(opt.batchSize,3,opt.targetSize,opt.targetSize)
  labels = torch.Tensor(opt.batchSize,opt.targetSize,opt.targetSize)

  for i = 1,opt.batchSize do
    inputs[i], labels[i] = trainHook(indices[i])
  end
  return inputs, labels
end


-- function to load val image
local function valHook(index)
  collectgarbage()
  local tmpInput = image.load(imagePath .. '/' .. valImages[index] .. '.jpg')
  local tmpLabel = image.load(labelPath .. '/' .. valImages[index] .. '.png')
  assert(tmpInput:size(3) == tmpLabel:size(3))
  assert(tmpInput:size(2) == tmpLabel:size(2))

  tmpLabel = tmpLabel * 255
  tmpLabel = tmpLabel + 1
  tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1

  local outImg = tmpInput
  local outLab = tmpLabel

  outImg = image.scale(tmpInput,opt.targetSize,opt.targetSize) --different size to have more space for random crops??
  outLab = image.scale(tmpLabel,opt.targetSize,opt.targetSize,'simple')

  -- mean/std
  for i=1,3 do -- channels
    if mean then outImg[{{i},{},{}}]:add(-mean[i]) 
    else error('no mean given')
    end
    --   if std then outImg[{{i},{},{}}]:div(std[i]) 
    --   else error('no std given')
    --   end
  end
  outImg = outImg:index(1,torch.LongTensor{3,2,1}) --rgb to bgr
  outImg = outImg * 255
  return outImg, outLab, valImages[index]
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
