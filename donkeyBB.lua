require 'image'
require 'mattorch'
require 'paths'

dataPath = '/data/DNN-common/DeconvPascal2012/VOC2012_TEST/ImageSets/Segmentation'
imagePath = '/data/DNN-common/DeconvPascal2012/VOC2012_TEST/JPEGImages'
boxPath = '/data/DNN-common/DeconvPascal2012/edgebox_cached/VOC2012_TEST'

local batchNumber = 0

--Load txt file for images
local testImagesFile = paths.concat(dataPath, 'testImages.t7')

if paths.filep(testImagesFile) then
  print('Loading testImage metadata from cache')
  testLoader = torch.load(testImagesFile)
  testImages = testLoader.testImages
  testLabels = testLoader.testLabels
else
  print('Creating testImage metadata')
--Load test
  testFile = assert(io.open(paths.concat(dataPath, "test.txt")))
  counter = 1
  testImages = {}

  for line in testFile:lines() do
    testImages[counter] = unpack(line:split(" "))
    -- if counter == 5 then break end  --to load only small subset of images
    counter = counter + 1
  end
  testFile:close()

  local testLoader = {}
  testLoader.testImages = testImages
  torch.save(testImagesFile, testLoader)
end

local meanStdFile = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation/meanStd.t7'
local meanstd = torch.load(meanStdFile)
mean = meanstd.mean
std = meanstd.std
print('Loaded mean and std from cache')

--------------------------------------------------------------------------------------------------------

function dimnarrow(x,sz,pad,dim)
  local xn = x
  for i=1,x:dim() do
    if i > dim then
      xn = xn:narrow(i,pad[i]+1,sz[i])
    end
  end
  return xn
end

function padzero(x,pad)
  local sz = x:size()
  for i=1,x:dim() do sz[i] = sz[i]+pad[i]*2 end
  local xx = x.new(sz):zero()
  local xn = dimnarrow(xx,x:size(),pad,-1)
  xn:copy(x)
  return xx
end

--------------------------------------------------------------------------------------------------------

-- function to load test image
local function testHook(index)
  collectgarbage()
  local tmpInput = image.load(imagePath .. '/' .. testImages[index] .. '.jpg')
  local tmpBoxes = mattorch.load(boxPath .. '/' .. testImages[index] .. '.mat')
  boxes = tmpBoxes.boxes_padded
  boxes = boxes:transpose(1,2)  --mattorch loads with swapped indices

  for i=1,3 do -- channels
    if mean then tmpInput[{{i},{},{}}]:add(-mean[i]) 
    else error('no mean given')
    end
    --  if std then tmpInput[{{i},{},{}}]:div(std[i]) 
    -- else error('no std given')
    -- end
  end
  tmpInput = tmpInput:index(1,torch.LongTensor{3,2,1}) --rgb to bgr
  tmpInput = tmpInput * 255

  pad = {}
  pad[1] = 0
  pad[2] = tmpInput:size(3)
  pad[3] = tmpInput:size(2)
  inputPad = padzero(tmpInput, pad) --padded_I

  boxImg = {}
  boxSize = {}

  count = 1
  for i = 1, boxes:size(1) do --opt.numBoxes do
    box_wd = boxes[i][3]-boxes[i][1]+1;
    box_ht = boxes[i][4]-boxes[i][2]+1;

    if math.min(box_wd, box_ht) > 112 then 
      boxImg[count] = image.crop(inputPad, boxes[i][1], boxes[i][2], boxes[i][3], boxes[i][4])
      boxImg[count] = image.scale(boxImg[count],224,224)
      boxSize[count] = boxes[i]
      count = count + 1
    end
    if count == opt.numBoxes then
      break
    end
  end
  
  return tmpInput, testImages[index], boxImg, boxSize
end

function loadTestBatch(indices)
  inputs = {}
  imNames = {}
  boxImgs = {}
  boxSizes = {}
  for i = 1,indices:size(1) do
    inputs[i], imNames[i], boxImgs[i], boxSizes[i] = testHook(indices[i])    
  end
  return inputs, imNames, boxImgs, boxSizes
end

-- input = batchSize x nClasses x H x W
-- target = batchSize x H x W