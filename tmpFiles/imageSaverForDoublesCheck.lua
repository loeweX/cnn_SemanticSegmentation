require 'image'

dataPath = '/data/DNN-common/DeconvPascal2012/imagesets/stage_1_train_imgset' --Stage 1 Training
imagePath = '/data/DNN-common/DeconvPascal2012/VOC2012'

-- a cache file of the training metadata (if doesnt exist, will be created)
trainImagesFile = paths.concat(dataPath, 'trainImages.t7')
valImagesFile = paths.concat(dataPath, 'valImages.t7')

trainLoader = torch.load(trainImagesFile)
trainImages = trainLoader.trainImages

valLoader = torch.load(valImagesFile)
valImages = valLoader.valImages

for i = 1,#trainImages  do  --estimate mean and std on a subset of the dataset
  tmp = image.load(paths.concat(imagePath, string.sub(trainImages[i],2,-1)))
  image.save('trainImg/' .. i .. '.jpg', tmp)
end

for i = 1,#valImages  do  --estimate mean and std on a subset of the dataset
  tmp = image.load(paths.concat(imagePath, string.sub(valImages[i],2,-1)))
  image.save('valImg/' .. i .. '.jpg', tmp)
end