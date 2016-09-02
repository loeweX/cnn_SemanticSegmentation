----------------------------------------------------------------------
-- classes + corresponding label numbers for the Pascal VOC Challenge
classes = { '0=background' , '1=aeroplane', '2=bicycle', '3=bird', '4=boat', '5=bottle', '6=bus',   
  '7=car', '8=cat', '9=chair', '10=cow', '11=diningtable', '12=dog', '13=horse', 
  '14=motorbike', '15=person', '16=potted plant', '17=sheep', '18=sofa', '19=train',  
  '20=tv/monitor'} -- all +1 !!
----------------------------------------------------------------------
print '==> configuring optimizer'

optimState = {
  learningRate = opt.learningRate,    
  learningRateDecay = 0.0,
  weightDecay = opt.weightDecay,
  momentum = opt.momentum, 
  dampening = 0.0
}

----------------------------------------------------------------------

print '==> defining training procedure'

local batchNumber = 0
local classAccuracy = torch.zeros(opt.numClasses)
local loss_epoch = 0
local confcounts = torch.zeros(opt.numClasses+1,opt.numClasses+1)
local iter_count = 0

function train()
  model:training()
  epoch = epoch or 1

  -- initliazing several parameters for this epoch
  classAccuracy = torch.zeros(opt.numClasses)
  loss_epoch = 0 
  confcounts = torch.zeros(opt.numClasses+1,opt.numClasses+1) 
  batchNumber = 0
  local tic = torch.tic()

  print(color.blue '==>'.." online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local indices = torch.randperm(numTrainImages):long():split(opt.batchSize)
  -- remove last element so that all the batches have equal size
  indices[#indices] = nil

  -- determine epochlength 
  if #indices < opt.epochLength then
    epochL = #indices
  else
    epochL = opt.epochLength
  end

  cutorch.synchronize()
  for t,v in ipairs(indices) do
    -- queue data loading jobs to donkeys
    donkeys:addjob(
      function() --load single batches
        local inputs, labels = loadTrainBatch(v)
        return inputs, labels
      end,
      -- train the loaded batch in the main thread
      trainBatch
    )
    if t == epochL then
      break
    end
  end

  donkeys:synchronize()
  cutorch.synchronize()

  print('\n')

  loss_epoch = loss_epoch / epochL
  train_acc = 0 
  train_loss = loss_epoch

  print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f   loss: %.2f',
      epoch, torch.toc(tic), loss_epoch))
  print('\n')

  collectgarbage()

  -- save the current model
  if epoch % opt.saveEpoch == 0 then
    if torch.type(model) == 'nn.DataParallelTable' then
      torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model:get(1))
    else
      torch.save(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model)
    end
  end
end


-- preallocate GPU inputs
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local tmp = 0

local parameters, gradParameters = model:getParameters()

-- trainBatch - used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
  cutorch.synchronize()
  collectgarbage()
  timer:reset()

  -- drop learning rate
  iter_count = iter_count + 1
  optimState.learningRate = opt.learningRate * math.pow(0.1, iter_count / opt.epoch_step)
  
  -- transfer over to GPU
  inputs = inputs or (opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
  labels = labels or torch.CudaTensor()

  inputs:resize(inputsCPU:size()):copy(inputsCPU)
  labels:resize(labelsCPU:size()):copy(labelsCPU)

  cutorch.synchronize()

  local outputs = torch.Tensor(opt.batchSize,opt.numClasses,opt.targetSize,opt.targetSize)

  local err = 0
  feval = function(x)
    model:zeroGradParameters()

    if opt.fullBatchDiv == 1 then -- train full batch at once
      local out = model:forward(inputs)
      outputs = out:float()
      err = criterion:forward(out, labels)
      local gradOutputs = criterion:backward(out, labels)
      model:backward(inputs, gradOutputs)
    else -- send several batches through the network, accumulate and average the gradients and train
      count = 0
      start = 1
      finish = opt.batchSize/ opt.fullBatchDiv 
      for i = 1, opt.fullBatchDiv do     
        local out = model:forward(inputs[{{start, finish},{},{},{}}])
        outputs[{{start, finish},{},{},{}}] = out:float()
        tmpErr = criterion:forward(out, labels[{{start, finish},{},{}}])
		
        -- check if loss was nan and keep only non-NaN values 
		--(happens for some reason for some batches, but training works still fine)
        if tmpErr == tmpErr then  
          err = err + tmpErr
          count = count + 1
        end
        local gradOutputs = criterion:backward(out, labels[{{start, finish},{},{}}])
        local gradInput = model:backward(inputs[{{start, finish},{},{},{}}], gradOutputs)    
      
        start = finish + 1
        finish = finish + opt.batchSize/ opt.fullBatchDiv         
        
        cutorch.synchronize()
      end
      gradParameters:mul(1/ opt.fullBatchDiv)
      if count > 0 then -- check again if accumulated loss is nan and keep the old value if it is the case
        err = err /count --/ opt.fullBatchDiv 
      else
        err = tmp
      end
      print(count)
    end
    return err, gradParameters
  end

  -- train with stochastic gradient descent
  optim.sgd(feval, parameters, optimState)

  tmp = err

-- DataParallelTable's syncParameters
  if model.needsSync then
    model:syncParameters()
  end

  cutorch.synchronize()

  batchNumber = batchNumber + 1
  loss_epoch = loss_epoch + err

  print(('Epoch: [%d][%d/%d]\tTime(s) %.3f  loss %.4f  LR %.4f  Net: %s'):format(
      epoch, batchNumber, math.floor(numTrainImages/opt.batchSize), timer:time().real, err, 
      optimState.learningRate, opt.netType))


  -- save some example input, output, label and output/label images for the first batch to be able to see the progress of the training
  if batchNumber == 1 then
    -- the 'correct' image has black pixels for all pixels that were labelled correctly by the network and white pixels for the wrong predictions
    correct = torch.zeros(opt.batchSize, opt.targetSize, opt.targetSize) + opt.numClasses + 1
    correct = correct:float()
    _,prediction_sorted = outputs:float():sort(2, true)
    prediction_sorted = prediction_sorted:float()
    labelCopy = labels:float()
    locsForCorrect = labelCopy:eq(prediction_sorted[{{},1,{},{}}])
    correct:maskedFill(locsForCorrect,1)--labelCopy)
    locsIrrelevant = labelCopy:eq(opt.numClasses + 1)
    correct:maskedFill(locsIrrelevant,1)

    imgCount = opt.batchSize < 16 and opt.batchSize or 16 -- save no more than 16 examples
    for i = 1,imgCount do
      -- get back to original input image (bgr->bgr and mean/std calculation)
      img = inputsCPU[i]:index(1,torch.LongTensor{3,2,1})
      img = img / 255
      for i=1,3 do -- channels
        if mean then img[{{i},{},{}}]:add(mean[i]) 
        else error('no mean given')
        end 
      end
	  -- save images for this epoch and overwrite the old results
      saveImages(img, prediction_sorted[{i,1,{},{}}], labels[i], correct[i], 'train', i)    
    end
	-- save one example for each epoch without overwriting the old one
    tmpStr = 'trEpoch' .. epoch
    saveImages(img, prediction_sorted[{imgCount,1,{},{}}], labels[imgCount], correct[imgCount], tmpStr, 1)  
  end
end
