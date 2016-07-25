-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------

require 'image'
require 'xlua'

dataPath = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation'
imagePath = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/myImages'

-- a cache file of the training metadata (if doesnt exist, will be created)
local testImagesFile = paths.concat(dataPath, 'valImages.t7') ------------------------------------------------------------------------------------
local meanStdFile = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation/meanStd.t7'

if paths.filep(testImagesFile) then
  print('Loading testImage metadata from cache')
  testLoader = torch.load(testImagesFile)
  testImages = testLoader.valImages ----------------------------------------------------------------------------------------------
else
  print('Creating testImage metadata')
--Load test
  testFile = assert(io.open(paths.concat(dataPath, "test.txt")))
  counter = 1
  testImages = {}

  for line in testFile:lines() do
    testImages[counter] = line
    -- if counter == 5 then break end  --to load only small subset of images
    counter = counter + 1
  end
  testFile:close()

  local testLoader = {}
  testLoader.testImages = testImages
  torch.save(testImagesFile, testLoader)
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

local sampleSize = {3, 224, 224}


-- function to load test image
local function testHook(index)
  collectgarbage()
  local tmpInput = image.load(imagePath .. '/' .. testImages[index] .. '.jpg')

  outImg = image.scale(tmpInput,224,224)

  -- mean/std
  for i=1,3 do -- channels
    if mean then outImg[{{i},{},{}}]:add(-mean[i]) 
    else error('no mean given')
    end
    if std then outImg[{{i},{},{}}]:div(std[i]) 
    else error('no std given')
    end
  end
  outImg = outImg:index(1,torch.LongTensor{3,2,1}) --rgb to bgr

  return outImg, testImages[index]
end

function loadTestBatch(indices)
  inputs = torch.Tensor(opt.batchSize,3,224,224)
  imNames = {}

  for i = 1,opt.batchSize do
    inputs[i], imNames[i] = testHook(indices[i])    
  end
  return inputs, imNames
end
