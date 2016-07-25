require 'mattorch' --loading Matlab-Data
require 'torch'   -- torch
require 'image'   -- for color transforms
require 'nn'      -- provides a normalization operator

print '==> loading dataset (this may take a while)'

-- We load the dataset from disk, and re-arrange it to be compatible
-- with Torch's representation. Matlab uses a column-major representation,
-- Torch is row-major, so we just have to transpose the data.

-- Note: the data, in X, is 4-d: the 1st dim indexes the samples, the 2nd
-- dim indexes the color channels (RGB), and the last two dims index the
-- height and width of the samples.

trainI = mattorch.load('/data/DNN-common/Pascal2012/VOCdevkit/trainImagesD.mat')  --D for Debugging, remove to have all images
trainI.trainImages = trainI.trainImages:transpose(2,4)
trainI.trainImages = trainI.trainImages:transpose(1,2)

trainL = mattorch.load('/data/DNN-common/Pascal2012/VOCdevkit/trainLabelsD.mat')
trainL.trainLabels = trainL.trainLabels:transpose(1,3)

trainData = {
  data = trainI.trainImages,
  labels = trainL.trainLabels,
  size = function() return trainI.trainImages:size(1) end
}

print('Size of the training data: ' .. trainData:size() .. ' images and ' .. trainData.labels:size(1) .. ' labels')

testI = mattorch.load('/data/DNN-common/Pascal2012/VOCdevkit/testImagesD.mat')
testI.testImages = testI.testImages:transpose(2,4)
testI.testImages = testI.testImages:transpose(1,2)

testL = mattorch.load('/data/DNN-common/Pascal2012/VOCdevkit/testLabelsD.mat')
testL.testLabels = testL.testLabels:transpose(1,3)

testData = {
  data = testI.testImages,
  labels = testL.testLabels,
  size = function() return testI.testImages:size(1) end
}

print('Size of the test data: ' .. testData:size() .. ' images and ' .. testData.labels:size(1) .. ' labels')

testData.labels = testData.labels + 1
testData.labels[testData.labels:gt(opt.numClasses+1)] = opt.numClasses+1

trainData.labels = trainData.labels + 1
trainData.labels[trainData.labels:gt(opt.numClasses+1)] = opt.numClasses+1

print '==> preprocessing data'

-- Preprocessing requires a floating point representation (the original
-- data is stored on bytes).
trainData.data = trainData.data:float()
testData.data = testData.data:float()


-- Name channels for convenience
channels = {'r','g','b'}

-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
print '==> preprocessing data: normalize each feature (channel) globally'
mean = {}
std = {}
for i,channel in ipairs(channels) do
  -- normalize each channel globally:
  mean[i] = trainData.data[{ {},i,{},{} }]:mean()
  std[i] = trainData.data[{ {},i,{},{} }]:std()
  trainData.data[{ {},i,{},{} }]:add(-mean[i])
  trainData.data[{ {},i,{},{} }]:div(std[i])
end

-- Normalize test data, using the training means/stds
for i,channel in ipairs(channels) do
  -- normalize each channel globally:
  testData.data[{ {},i,{},{} }]:add(-mean[i])
  testData.data[{ {},i,{},{} }]:div(std[i])
end

-- save mean and std to make future image classification possible
local filename = paths.concat(opt.save, 'trainMean.t7')
os.execute('mkdir -p ' .. sys.dirname(filename))
torch.save(filename, mean)
local filename = paths.concat(opt.save, 'trainStd.t7')
os.execute('mkdir -p ' .. sys.dirname(filename))
torch.save(filename, std)

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

for i,channel in ipairs(channels) do
  trainMean = trainData.data[{ {},i }]:mean()
  trainStd = trainData.data[{ {},i }]:std()

  testMean = testData.data[{ {},i }]:mean()
  testStd = testData.data[{ {},i }]:std()

  print('training data, '..channel..'-channel, mean: ' .. trainMean)
  print('training data, '..channel..'-channel, standard deviation: ' .. trainStd)

  print('test data, '..channel..'-channel, mean: ' .. testMean)
  print('test data, '..channel..'-channel, standard deviation: ' .. testStd)
end

----------------------------------------------------------------------

-- Visualization is quite easy, using itorch.image().

if itorch then
  print '==> visualizing data'

  print('Training Data')
  first10Images = trainData.data[{ {1,10} }]
  first10Labels = trainData.label[{ {1,10} }]

  itorch.image(first10Images)
  itorch.image(first10Labels)
  
  print('Test Data')
  first10Images = testData.data[{ {1,10} }]
  first10Labels = testData.label[{ {1,10} }]

  itorch.image(first10Images)
  itorch.image(first10Labels)
else
  print("==> For visualization, run this script in an itorch notebook")
end
