require 'torch'   -- torch
require 'xlua'    -- xlua provides useful tools, like progress bars
require 'optim'   -- an optimization package, for online and batch methods
require 'image'
require 'cutorch'

local c = require 'trepl.colorize'

----------------------------------------------------------------------
-- train on GPU
model:cuda()
model = makeDataParallel(model, opt.nGPU) -- defined in util.lua - training on multiple GPUs

criterion:cuda()

----------------------------------------------------------------------
-- classes
classes = { '0=background' , '1=aeroplane', '2=bicycle', '3=bird', '4=boat', '5=bottle', '6=bus',             
  '7=car', '8=cat', '9=chair', '10=cow', '11=diningtable', '12=dog', '13=horse', 
  '14=motorbike', '15=person', '16=potted plant', '17=sheep', '18=sofa', '19=train',        
  '20=tv/monitor'} -- alle +1 !!

parameters,gradParameters = model:getParameters()

----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
  learningRate = opt.learningRate,    
  learningRateDecay = 0.0,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum, 
  dampening = 0.0
  --learningRateDecay = 1e-7
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

optimMethod = optim.sgd

----------------------------------------------------------------------
print '==> defining training procedure'


function train()
  model:training()
  epoch = epoch or 1
  local batchNumber = 0
  local truePosAll_epoch = 0
  local classAccuracy = torch.zeros(opt.numClasses)
  local loss_epoch = 0

  -- drop learning rate every "epoch_step" epochs
  if epoch % opt.epoch_step == 0 then 
    optimState.learningRate = optimState.learningRate * 0.5 
  end

  print(c.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  indices = torch.randperm(#trainImages):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  cutorch.synchronize()

  local tic = torch.tic()
  for t,v in ipairs(indices) do
    batchNumber = batchNumber + 1
    cutorch.synchronize()

    inputs = torch.CudaTensor(opt.batchSize,3,224,224)
    labels = torch.CudaTensor(opt.batchSize,opt.targetSize,opt.targetSize)

    cropR = torch.rand(opt.batchSize)
    flipR = torch.rand(opt.batchSize)

    for i = 1,opt.batchSize do
      tmpInput = image.load(paths.concat(imagePath, string.sub(trainImages[v[i]],2,-1)))
      tmpLabel = image.load(paths.concat(imagePath, string.sub(trainLabels[v[i]],2,-1)))

      for j = 1,3 do    
        tmpInput[j] = (tmpInput[j] - image_mean[j]) / image_std[j]
      end

      tmpLabel = tmpLabel * 255
      tmpLabel = tmpLabel + 1
      tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1

      tmpInput = image.scale(tmpInput,250,250,'simple')
      tmpLabel = image.scale(tmpLabel,250,250,'simple')

      -- flip half of the images and choose part to be cropped from image --> 10 different pictures from one
    
      if flipR[i] < 0.5 then
        tmpInput = image.hflip(tmpInput)      
        tmpLabel = image.hflip(tmpLabel)      
      end
      if cropR[i] < 0.2 then
        inputs[i] = image.crop(tmpInput, 'c', 224, 224)
        labels[i] = image.crop(tmpLabel, 'c', 224, 224)
      elseif cropR[i] < 0.4 then
        inputs[i] = image.crop(tmpInput, 'tl', 224, 224)
        labels[i] = image.crop(tmpLabel, 'tl', 224, 224)
      elseif cropR[i] < 0.6 then
        inputs[i] = image.crop(tmpInput, 'tr', 224, 224)
        labels[i] = image.crop(tmpLabel, 'tr', 224, 224)
      elseif cropR[i] < 0.8 then
        inputs[i] = image.crop(tmpInput, 'bl', 224, 224)
        labels[i] = image.crop(tmpLabel, 'bl', 224, 224)
      else
        inputs[i] = image.crop(tmpInput, 'br', 224, 224)
        labels[i] = image.crop(tmpLabel, 'br', 224, 224)
      end 
  end

  local err, outputs
  local feval = function(x)
    model:zeroGradParameters()
    gradParameters:zero()
    outputs = model:forward(inputs)  
    err = criterion:forward(outputs, labels)
    local gradOutputs = criterion:backward(outputs, labels)     
    model:backward(inputs, gradOutputs)

    return err, gradParameters
  end

  optim.sgd(feval, parameters, optimState)

  if model.needsSync then
    model:syncParameters()   
  end

  cutorch.synchronize()

  loss_epoch = loss_epoch + err

-- top-1 error
  truePosAll = 0
  truePos = torch.zeros(opt.numClasses)
  counter = 0
  correct = torch.zeros(opt.batchSize,opt.targetSize,opt.targetSize) + opt.numClasses + 1
  count_predicted = torch.zeros(opt.numClasses)
  do
    local _,prediction_sorted = outputs:float():sort(2, true) -- descending
    labels = labels:float()
    for i=1,opt.batchSize do
      for j=1,opt.targetSize do
        for k=1,opt.targetSize do
          if labels[{i,j,k}] < opt.numClasses + 1 then
            counter = counter + 1
            count_predicted[prediction_sorted[{i,1,j,k}]] = count_predicted[prediction_sorted[{i,1,j,k}]] + 1
            if prediction_sorted[{i,1,j,k}] == labels[{i,j,k}] then
              truePosAll = truePosAll + 1
              correct[{i,j,k}] = labels[{i,j,k}] --show correct points in image
              truePos[labels[{i,j,k}]] = truePos[labels[{i,j,k}]] + 1
            end
          end
        end
      end
    end
    count_label = torch.zeros(opt.numClasses)
    single_accuracy = torch.zeros(opt.numClasses)

    for i = 1, opt.numClasses do
      count_label[i] = labels:eq(i):sum()

      if (count_predicted[i] + count_label[i] - truePos[i]) == 0 then
        single_accuracy[i] = 1
      else
        single_accuracy[i] = truePos[i]  / (count_predicted[i] + count_label[i] - truePos[i])
      end
    end
  end

  collectgarbage()
  cutorch.synchronize()

  truePosAll = truePosAll * 100 / counter  -- to get %
  truePosAll_epoch = truePosAll_epoch + truePosAll
  classAccuracy = classAccuracy + single_accuracy

  if batchNumber % 1 == 0 then    
    print(('Epoch: [%d][%d/%d]\t Err %.4f Top1-%%: %.2f IoU: %.2f'):format(
        epoch, batchNumber, #indices, err, truePosAll, single_accuracy:mean()*100))
  end

  print('\n')

  if t == #indices then
    imgCount = opt.batchSize < 16 and opt.batchSize or 16
    local _,prediction_sorted = outputs:float():sort(2, true)
    for i = 1,imgCount do
      saveImages(inputs[i], prediction_sorted[{i,1,{},{}}], labels[i], correct[i], 'train', i)    end
    if epoch % 2 == 0 then
      tmpStr = 'trEpoch' .. epoch
      saveImages(inputs[1], prediction_sorted[{1,1,{},{}}], labels[1], correct[1], tmpStr, 1)  
    end
  end

end

truePosAll_epoch = truePosAll_epoch / #indices -- #indices = number of batches per epoch
classAccuracy = classAccuracy / #indices
loss_epoch = loss_epoch / #indices

train_acc = classAccuracy:mean()*100 -- truePosAll_epoch
train_loss = loss_epoch

print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
    .. 'average loss: %.2f \t '
    .. 'accuracy(%%):\t top-1 %.2f\t  IoU: %.2f',
    epoch, torch.toc(tic), loss_epoch, truePosAll_epoch, classAccuracy:mean()*100))
print('\n')

collectgarbage()

if epoch % opt.saveEpoch == 0 then
  saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:clearState()) -- defined in util.lua
  torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end
end