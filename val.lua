print '==> defining validating procedure'

local batchNumber = 0
local loss_epoch = 0
local confcounts = torch.zeros(opt.numClasses+1,opt.numClasses+1)

local meanStdFile = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation/meanStd.t7'
local meanstd = torch.load(meanStdFile)
mean = meanstd.mean
std = meanstd.std
print('Loaded mean and std from cache')


function val()
  model:evaluate()  

  batchNumber = 0
  loss_epoch = 0
  confcounts = torch.zeros(opt.numClasses+1,opt.numClasses+1)

  local tic = torch.tic()

  if opt.fullBatchDiv ~= 'none' then
    opt.batchSize = opt.batchSize / opt.batchDifVal
    print('BatchSize for validation: ' .. opt.batchSize)
  end

  indices = torch.randperm(numValImages):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  print('validating: ')

  if #indices < opt.epochLengthVal then
    epochL = #indices
  else
    epochL = opt.epochLengthVal
  end

  cutorch.synchronize()
  for t,v in ipairs(indices) do
    -- queue jobs to data-workers
    donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function() --load single batches
        local inputs, labels, imNames = loadValBatch(v)
        return inputs, labels, imNames
      end,
      -- the end callback (runs in the main thread)
      valBatch
    )
    xlua.progress(t, epochL)
    if t == opt.epochLengthVal then
      break
    end
  end

  donkeys:synchronize()
  cutorch.synchronize()

  print('\n')
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

  accLogger:add{train_acc, val_acc}
  accLogger:style{'-','-'}
  accLogger:plot()

  lossLogger:add{train_loss, val_loss}
  lossLogger:style{'-','-'}
  lossLogger:plot()

  writeReport(single_accuracy)

  if opt.fullBatchDiv ~= 'none' then
    opt.batchSize = opt.batchSize * opt.batchDifVal
  end

  collectgarbage()
end


-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local tmp = 0

--local parameters, gradParameters = model:getParameters()

function valBatch(inputsCPU, labelsCPU, imNames)
  cutorch.synchronize()
  collectgarbage()

  -- transfer over to GPU
-- inputs = inputsCPU:cuda()
--  labels = labelsCPU:cuda()
  inputs = inputs or (opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
  labels = labels or torch.CudaTensor()

  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  cutorch.synchronize()

  outputs = model:forward(inputs)
  err = criterion:forward(outputs, labels)

  if err == err then

  else
    err = tmp
  end
  tmp = err

  cutorch.synchronize()

  batchNumber = batchNumber + 1
  loss_epoch = loss_epoch + err
  correct = torch.ones(opt.batchSize, opt.targetSize, opt.targetSize)
  outLabel = opt.numClasses + 1
  -- confcounts = torch.zeros(outLabel,outLabel)
  --do
  _,prediction_sorted = outputs:float():sort(2, true) -- descending
  assert(prediction_sorted:max() < outLabel)
  tmpOut = prediction_sorted[{{},1,{},{}}]:float()
  tmpLab = labels:float()
  tmpLab = tmpLab:float()
  correct = torch.zeros(opt.batchSize, opt.targetSize,opt.targetSize) + opt.numClasses + 1

  assert(prediction_sorted:size(3) == tmpLab:size(2) and prediction_sorted:size(4) == tmpLab:size(3))
  tmpOut = tmpOut:long()
  locsToConsider = tmpLab:ne(outLabel)
  locsToConsider = locsToConsider:byte()

  tmp = tmpLab:long()
  sumImg = tmp:add(tmpOut*outLabel) + 1
  tmpImg = sumImg:maskedSelect(locsToConsider)
  tmpImg = tmpImg:float()
  hs = torch.histc(tmpImg, outLabel*outLabel+1, 1, outLabel*outLabel+1)
  confcounts = confcounts:add(torch.reshape(hs[{{1,outLabel*outLabel}}],outLabel,outLabel))

  if batchNumber == 1 then
    correct = correct:float()
    tmpOut = tmpOut:float()
    locsForCorrect = tmpLab:eq(tmpOut)
    correct:maskedFill(locsForCorrect,1)--labelCopy)

    locsIrrelevant = tmpLab:eq(opt.numClasses + 1)
    correct:maskedFill(locsIrrelevant,1)

    imgCount = opt.batchSize < 16 and opt.batchSize or 16 -- for pretty printing
    for i = 1,imgCount do
      --calculate back to original image (bgr->bgr and mean/std calculation)
      img = inputsCPU[i]:index(1,torch.LongTensor{3,2,1})
      img = img / 255
      for i=1,3 do -- channels
        --  if std then img[{{i},{},{}}]:mul(std[i]) 
        --  else error('no std given')
        --  end
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