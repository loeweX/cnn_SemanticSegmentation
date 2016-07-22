local batchNumber
local truePosAll, loss_epoch
local timer = torch.Timer()

function test()
  print('==> doing epoch on validation data:')

  batchNumber = 0
  local truePosAll_epoch = 0
  local classAccuracy = torch.zeros(opt.numClasses)
  cutorch.synchronize()

  local tic = torch.tic()

  -- set the dropouts to evaluate mode
  model:evaluate()  
  truePosAll = 0
  loss_epoch = 0

  indices = torch.randperm(#valImages):long():split(opt.batchSize)
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
      tmpInput = image.load(paths.concat(imagePath, string.sub(valImages[v[i]],2,-1)))
      tmpLabel = image.load(paths.concat(imagePath, string.sub(valLabels[v[i]],2,-1)))

      for j = 1,3 do    
        tmpInput[j] = (tmpInput[j] - image_mean[j]) / image_std[j]
      end

      tmpLabel = tmpLabel * 255
      tmpLabel = tmpLabel + 1
      tmpLabel[tmpLabel:gt(opt.numClasses+1)] = opt.numClasses + 1

      tmpInput = image.scale(tmpInput,250,250,'simple')
      tmpLabel = image.scale(tmpLabel,250,250,'simple')

      inputs[i] = image.crop(tmpInput, 'c', 224, 224)
      labels[i] = image.crop(tmpLabel, 'c', 224, 224)
    end

    outputs = model:forward(inputs)
    local err = criterion:forward(outputs, labels)

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

    loss_epoch = loss_epoch + err  
    truePosAll = truePosAll * 100 / counter  -- to get %
    truePosAll_epoch = truePosAll_epoch + truePosAll
    classAccuracy = classAccuracy + single_accuracy

    if t == #indices then
      imgCount = opt.batchSize < 16 and opt.batchSize or 16
      local _,prediction_sorted = outputs:float():sort(2, true)
      for i = 1,imgCount do
        saveImages(inputs[i], prediction_sorted[{i,1,{},{}}], 
          labels[i], correct[i], 'val', i)      
      end
      if epoch % 2 == 0 then
        tmpStr = 'teEpoch' .. epoch
        saveImages(inputs[1], prediction_sorted[{1,1,{},{}}], labels[1], correct[1], tmpStr, 1) 
      end
    end
  end

  cutorch.synchronize()

  truePosAll_epoch = truePosAll_epoch / #indices -- #indices = number of batches per epoch
  classAccuracy = classAccuracy / #indices
  loss_epoch = loss_epoch / #indices

  print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
      .. 'average loss: %.2f \t '
      .. 'accuracy [Center](%%):\t top-1 %.2f\t IoU: %.2f',
      epoch, torch.toc(tic), loss_epoch, truePosAll_epoch, classAccuracy:mean()*100))

  print('\n')

--for logger
  test_acc = classAccuracy:mean()*100
  test_loss = loss_epoch

  accLogger:add{train_acc, test_acc}
  accLogger:style{'-','-'}
  accLogger:plot()

  lossLogger:add{train_loss, test_loss}
  lossLogger:style{'-','-'}
  lossLogger:plot()

  writeReport(classAccuracy)

end 
