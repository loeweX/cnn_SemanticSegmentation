require 'xlua'
dataPath = '/data/DNN-common/DeconvPascal2012/imagesets/stage_2_train_imgset'

local trainImagesFile = paths.concat(dataPath, 'trainValImages.t7')
local valImagesFile = paths.concat(dataPath, 'valImages.t7')

print('Loading trainImage metadata from cache')
trainLoader = torch.load(trainImagesFile)
trainImages = trainLoader.trainImages
trainLabels = trainLoader.trainLabels
cropX1 = trainLoader.cropX1
cropY1 = trainLoader.cropY1
cropX2 = trainLoader.cropX2
cropY2 = trainLoader.cropY2

valLoader = torch.load(valImagesFile)
valImages = valLoader.valImages
valLabels = valLoader.valLabels
VcropX1 = valLoader.cropX1
VcropY1 = valLoader.cropY1
VcropX2 = valLoader.cropX2
VcropY2 = valLoader.cropY2

checker = {}

newTrainImages = {}
newTrainLabels = {}
newX1 = {}
newY1 = {}
newX2 = {}
newY2 = {}

----------------------IMAGES
hash = {}
for t,v in ipairs(valImages) do
 -- hash[v .. VcropX1[t] .. VcropX2[t] .. VcropY1[t] .. VcropY2[t]] = true
  hash[v] = true
  xlua.progress(t, #valImages)
end


for t,v in ipairs(trainImages) do
 -- if (not hash[v .. cropX1[t] .. cropX2[t] .. cropY1[t] .. cropY2[t]]) then
  if (not hash[v]) then
    newTrainImages[#newTrainImages+1] = v
    newTrainLabels[#newTrainLabels+1] = trainLabels[t]
    newX1[#newX1+1] = cropX1[t]
    newX2[#newX2+1] = cropX2[t]
    newY1[#newY1+1] = cropY1[t]
    newY2[#newY2+1] = cropY2[t]
  --hash[v .. VcropX1[t] .. VcropX2[t] .. VcropY1[t] .. VcropY2[t]] = true
   hash[v] = true
 end
  xlua.progress(t, #trainImages)
end
print(#newTrainImages)

--[[
----------------------LABELS
local hash = {}
for t,v in ipairs(valLabels) do
  hash[v] = true
  xlua.progress(t, #valLabels)
end

for t,v in ipairs(trainLabels) do
  if (not hash[v]) then
    newTrainLabels[#newTrainLabels+1] = v
  --  hash[v] = true
  end
  xlua.progress(t, #trainLabels)
end

print(#newTrainLabels) ]]--

local trainLoader = {}
trainLoader.trainImages = newTrainImages
trainLoader.trainLabels = newTrainLabels
trainLoader.cropX1 = newX1
trainLoader.cropY1 = newY1
trainLoader.cropX2 = newX2
trainLoader.cropY2 = newY2
--torch.save(paths.concat(dataPath, 'trainImages.t7'), trainLoader)