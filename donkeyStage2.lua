require 'image'
require 'xlua'

dataPath = '/data/DNN-common/DeconvPascal2012/imagesets/stage_2_train_imgset' --Stage 2 Training
imagePath = '/data/DNN-common/DeconvPascal2012/VOC2012'

-- a cache file of the training metadata (if doesnt exist, will be created)
if opt.trainValSplit then
  trainImagesFile = paths.concat(dataPath, 'trainImages.t7')
else
  trainImagesFile = paths.concat(dataPath, 'trainValImages.t7')
end
local valImagesFile = paths.concat(dataPath, 'valImages.t7')
local meanStdFile = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation/meanStd.t7'

trainImages = {}
trainLabels = {}

if paths.filep(trainImagesFile) then
  print('Loading trainImage metadata from cache')
  trainLoader = torch.load(trainImagesFile)
  trainImages = trainLoader.trainImages
  trainLabels = trainLoader.trainLabels
  cropX1 = trainLoader.cropX1
  cropY1 = trainLoader.cropY1
  cropX2 = trainLoader.cropX2
  cropY2 = trainLoader.cropY2
else
  print('Creating trainImage metadata')
  --Load Train
  trainFile = assert(io.open(paths.concat(dataPath, "train.txt")))
  counter = 1
  trainImages = {}
  trainLabels = {}
  cropX1 = {}
  cropY1 = {}
  cropX2 = {}
  cropY2 = {}

  for line in trainFile:lines() do
    trainImages[counter], trainLabels[counter], cropX1[counter], cropY1[counter], cropX2[counter], cropY2[counter] = unpack(line:split(" "))
    -- if counter == 5 then break end  --to load only small subset of images
    counter = counter + 1
  end
  trainFile:close()

  trainLoader = {}
  trainLoader.trainImages = trainImages
  trainLoader.trainLabels = trainLabels
  trainLoader.cropX1 = cropX1
  trainLoader.cropY1 = cropY1
  trainLoader.cropX2 = cropX2
  trainLoader.cropY2 = cropY2
  torch.save(trainImagesFile, trainLoader)
end


if paths.filep(valImagesFile) then
  print('Loading valImage metadata from cache')
  valLoader = torch.load(valImagesFile)
  valImages = valLoader.valImages
  valLabels = valLoader.valLabels
  valCropX1 = valLoader.cropX1
  valCropY1 = valLoader.cropY1
  valCropX2 = valLoader.cropX2
  valCropY2 = valLoader.cropY2
else
  print('Creating valImage metadata')
--Load val
  valFile = assert(io.open(paths.concat(dataPath, "val.txt")))
  counter = 1
  valImages = {}
  valLabels = {}
  cropX1 = {}
  cropY1 = {}
  cropX2 = {}
  cropY2 = {}

  for line in valFile:lines() do
    valImages[counter], valLabels[counter], cropX1[counter], cropY1[counter], cropX2[counter], cropY2[counter] = unpack(line:split(" "))
    -- if counter == 5 then break end  --to load only small subset of images
    counter = counter + 1
  end
  valFile:close()

  local valLoader = {}
  valLoader.valImages = valImages
  valLoader.valLabels = valLabels
  valLoader.cropX1 = cropX1
  valLoader.cropY1 = cropY1
  valLoader.cropX2 = cropX2
  valLoader.cropY2 = cropY2
  torch.save(valImagesFile, valLoader)
end

local meanstd = torch.load(meanStdFile)
mean = meanstd.mean
std = meanstd.std
print('Loaded mean and std from cache')


collectgarbage()

-----------------------------------------------------------------------------------------------------------------

local function copyMakeBorder(img, top, bottom, left, right, value)
  --[[ img = tmpInput
  top = pad_Y1
  bottom = pad_Y2
  left = pad_X1
  right = pad_X2
  value = 0 ]]--
  newHeight = img:size(2) + top + bottom
  newWidth = img:size(3) + left + right
  newImg = torch.zeros(img:size(1), newHeight, newWidth) + value
  newImg[{{},{bottom+1, bottom+img:size(2)},{left+1,left+img:size(3)}}] = img
  return newImg  
end

-- function to load train image, jitter it appropriately (random crops etc.)
local function trainHook(index)
  collectgarbage()
  tmpInput = image.load(paths.concat(imagePath, string.sub(trainImages[index],2,-1)))
  tmpLabel = image.load(paths.concat(imagePath, string.sub(trainLabels[index],2,-1)))

  assert(tmpInput:size(3) == tmpLabel:size(3))
  assert(tmpInput:size(2) == tmpLabel:size(2))

  tmpLabel = tmpLabel * 255
  tmpLabel = tmpLabel + 1
  tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1

  -----------------------------------------------------------------------------
  -- crop the image using the given values (cropX...)
  
  pad_X1 = math.max(0, -cropX1[index])
  pad_Y1 = math.max(0, -cropY1[index])

  pad_X2 = math.max(0, cropX2[index] - tmpInput:size(3) + 1)
  pad_Y2 = math.max(0, cropY2[index] - tmpInput:size(2) + 1)

  if pad_X1 > 0 or pad_Y1 > 0 or pad_X2 > 0 or pad_Y2 > 0 then
    tmpInput = copyMakeBorder(tmpInput, pad_Y1, pad_Y2, pad_X1, pad_X2, 0)
    tmpLabel = copyMakeBorder(tmpLabel, pad_Y1, pad_Y2, pad_X1, pad_X2, opt.numClasses+1)
  end

  x1 = cropX1[index] + pad_X1 + 1
  x2 = cropX2[index] + pad_X1 + 1
  y1 = cropY1[index] + pad_Y1 + 1
  y2 = cropY2[index] + pad_Y1 + 1

  assert(x1 > -1)
  assert(y1 > -1)
  assert(x2 <= tmpInput:size(3), 'mismatched sizes: ' .. x2 .. ' : ' .. tmpInput:size(3))
  assert(y2 <= tmpInput:size(2), 'mismatched sizes: ' .. y2 .. ' : ' .. tmpInput:size(2))

  tmpInput = image.crop(tmpInput, x1, y1, x2, y2)
  tmpLabel = image.crop(tmpLabel, x1, y1, x2, y2)

  -----------------------------------------------------------------------------

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
  local tmpInput = image.load(paths.concat(imagePath, string.sub(valImages[index],2,-1)))
  local tmpLabel = image.load(paths.concat(imagePath, string.sub(valLabels[index],2,-1)))

  assert(tmpInput:size(3) == tmpLabel:size(3))
  assert(tmpInput:size(2) == tmpLabel:size(2))

  tmpLabel = tmpLabel * 255
  tmpLabel = tmpLabel + 1
  tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1
  
  ---------------------------------------------------
  pad_X1 = math.max(0, -valCropX1[index])
  pad_Y1 = math.max(0, -valCropY1[index])

  pad_X2 = math.max(0, valCropX2[index] - tmpInput:size(3) + 1)
  pad_Y2 = math.max(0, valCropY2[index] - tmpInput:size(2) + 1)

  if pad_X1 > 0 or pad_Y1 > 0 or pad_X2 > 0 or pad_Y2 > 0 then
    tmpInput = copyMakeBorder(tmpInput, pad_Y1, pad_Y2, pad_X1, pad_X2, 0)
    tmpLabel = copyMakeBorder(tmpLabel, pad_Y1, pad_Y2, pad_X1, pad_X2, opt.numClasses+1)
  end

  x1 = valCropX1[index] + pad_X1 + 1
  x2 = valCropX2[index] + pad_X1 + 1
  y1 = valCropY1[index] + pad_Y1 + 1
  y2 = valCropY2[index] + pad_Y1 + 1

  assert(x1 > -1)
  assert(y1 > -1)
  assert(x2 <= tmpInput:size(3), 'mismatched sizes: ' .. x2 .. ' : ' .. tmpInput:size(3))
  assert(y2 <= tmpInput:size(2), 'mismatched sizes: ' .. y2 .. ' : ' .. tmpInput:size(2))

  tmpInput = image.crop(tmpInput, x1, y1, x2, y2)
  tmpLabel = image.crop(tmpLabel, x1, y1, x2, y2)
  
---------------------------------------------------
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
