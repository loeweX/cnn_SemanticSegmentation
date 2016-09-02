print '==> defining validating procedure'

local batchNumber = 0
local loss_epoch = 0
local confcounts = torch.zeros(opt.numClasses+1,opt.numClasses+1)

function val()
  model:evaluate()  

  -- initiliaze some values for this epoch
  batchNumber = 0
  loss_epoch = 0
  confcounts = torch.zeros(opt.numClasses+1,opt.numClasses+1)

  local tic = torch.tic()

  indices = torch.randperm(numValImages):long():split(opt.batchSizeVal)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  print('validating: ')

  -- determine epoch length
  if #indices < opt.epochLengthVal then
    epochL = #indices
  else
    epochL = opt.epochLengthVal
  end

  cutorch.synchronize()
  for t,v in ipairs(indices) do
    donkeys:addjob(
      function() --load single batches
        local inputs, labels, imNames = loadValBatch(v)
        return inputs, labels, imNames
      end,
      -- test their performance in the main thread
      valBatch
    )
    xlua.progress(t, epochL) -- print progress bar in terminal
    if t == opt.epochLengthVal then
      break
    end
  end

  donkeys:synchronize()
  cutorch.synchronize()

  print('\n')
  
  -- calculate intersection over union final score for this epoch
  single_accuracy = torch.zeros(opt.numClasses)
  for i = 2, opt.numClasses+1 do
    label = confcounts[{i,{}}]:sum()
    result = confcounts[{{},i}]:sum()
    truePos = confcounts[{i,i}]
    single_accuracy[i-1] = 100 * truePos / (label + result - truePos)
  end

  loss_epoch = loss_epoch / epochL
  val_acc = single_accuracy:mean() 
  val_loss = loss_epoch

  print(string.format('Epoch: [%d][VALIDATING SUMMARY] Total Time(s): %.2f   loss: %.2f   IoU-%%: %.2f',
      epoch, torch.toc(tic), loss_epoch, single_accuracy:mean()))
  print('\n')

  print('Accuracies for all classes: ')
  for i = 1,opt.numClasses do
    print(classes[i] .. ': \t \t ' .. single_accuracy[i])
  end
  print('\n')

  -- write information in logger files and update report.html
  accLogger:add{train_acc, val_acc}
  accLogger:style{'-','-'}
  accLogger:plot()

  lossLogger:add{train_loss, val_loss}
  lossLogger:style{'-','-'}
  lossLogger:plot()

  writeReport(single_accuracy)

  collectgarbage()
end


-- preallocate GPU inputs
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local tmpErr = 0

--local parameters, gradParameters = model:getParameters()

function valBatch(inputsCPU, labelsCPU, imNames)
  cutorch.synchronize()
  collectgarbage()

  -- transfer over to GPU
  inputs = inputs or (opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
  labels = labels or torch.CudaTensor()

  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  cutorch.synchronize()

  -- send batch through network
  outputs = model:forward(inputs)
  err = criterion:forward(outputs, labels)

  if err == err then -- if resulting err is NaN, keep old value 
  else
    err = tmpErr
  end
  tmpErr = err

  cutorch.synchronize()

  batchNumber = batchNumber + 1
  loss_epoch = loss_epoch + err
  
  -- initiliaze some values for accuracy calculation
  correct = torch.ones(opt.batchSizeVal, opt.targetSize, opt.targetSize)
  outLabel = opt.numClasses + 1
  _,prediction_sorted = outputs:float():sort(2, true) -- descending
  assert(prediction_sorted:max() < outLabel)
  tmpOut = prediction_sorted[{{},1,{},{}}]:float()
  tmpLab = labels:float()
  tmpLab = tmpLab:float()
  correct = torch.zeros(opt.batchSizeVal, opt.targetSize,opt.targetSize) + opt.numClasses + 1

  assert(prediction_sorted:size(3) == tmpLab:size(2) and prediction_sorted:size(4) == tmpLab:size(3))
  tmpOut = tmpOut:long()
  locsToConsider = tmpLab:ne(outLabel) -- label 'outLabel' does not contribute to accuracy calculation
  locsToConsider = locsToConsider:byte()

  -- calculate/update confusion matrix (confcounts)
  tmp = tmpLab:long()
  sumImg = tmp:add(tmpOut*outLabel) + 1
  tmpImg = sumImg:maskedSelect(locsToConsider)
  tmpImg = tmpImg:float()
  hs = torch.histc(tmpImg, outLabel*outLabel+1, 1, outLabel*outLabel+1)
  confcounts = confcounts:add(torch.reshape(hs[{{1,outLabel*outLabel}}],outLabel,outLabel))

  -- save some examples for the first batch (similar as in train.lua)
  if batchNumber == 1 then
    correct = correct:float()
    tmpOut = tmpOut:float()
    locsForCorrect = tmpLab:eq(tmpOut)
    correct:maskedFill(locsForCorrect,1)

    locsIrrelevant = tmpLab:eq(opt.numClasses + 1)
    correct:maskedFill(locsIrrelevant,1)

    imgCount = opt.batchSizeVal < 16 and opt.batchSizeVal or 16 -- save at most 16 examples
    for i = 1,imgCount do
      --get back to original input image (bgr->bgr and mean/std calculation)
      img = inputsCPU[i]:index(1,torch.LongTensor{3,2,1})
      img = img / 255
      for i=1,3 do -- channels
        if mean then img[{{i},{},{}}]:add(mean[i]) 
        else error('no mean given')
        end 
      end
      saveImages(img, tmpOut[{i,{},{}}], tmpLab[i], correct[i], 'val', i)    
    end
    tmpStr = 'teEpoch' .. epoch
    saveImages(img, tmpOut[{imgCount,{},{}}], tmpLab[imgCount], correct[imgCount], tmpStr, 1)  
  end
end