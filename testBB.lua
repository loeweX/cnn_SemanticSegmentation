require 'xlua'
local matio = require 'matio'

print '==> defining testing procedure'

local meanStdFile = opt.meanFilePath
local meanstd = torch.load(meanStdFile)
mean = meanstd.mean
std = meanstd.std
print('Loaded mean and std from cache')
paths.mkdir(opt.save .. '/finalImages')

local timer = torch.Timer()


function test()
  model:evaluate()  

  batchNumber = 0
  local tic = torch.tic()

  indices = torch.randperm(numTestImages):long():split(opt.batchSize)

  print('testing: ')

  epochL = #indices

  cutorch.synchronize()
  for t,v in ipairs(indices) do
    donkeys:addjob(
      function() --load single batches
        inputs, imNames, boxImgs, boxSizes = loadTestBatch(v)
        return inputs, imNames, boxImgs, boxSizes
      end,
      testBatch
    )
  end

  donkeys:synchronize()
  cutorch.synchronize()

  print(string.format('[TESTING SUMMARY] Total Time(s): %.2f', torch.toc(tic)))
  print('\n')

  writeReportTest()

  collectgarbage()
end


-- preallocate GPU inputs
local boxes = torch.CudaTensor()

function testBatch(inputsCPU, imNames, boxesCPU, boxSizes)
  cutorch.synchronize()
  collectgarbage()
  -- transfer over to GPU
  timer:reset()
  final_out = {}

  -- very slow implementation that sends every bounding box image seperately to the network (faster implementation in testBB_fast.lua, but not fully tested)
  for inpC = 1, #inputsCPU do
    local padded_output = torch.zeros(opt.numClasses,inputsCPU[inpC]:size(2) + 2 * inputsCPU[inpC]:size(3),inputsCPU[inpC]:size(3) + 2 * inputsCPU[inpC]:size(2))
    padded_output[1] = 0.00001
    
    local input = torch.CudaTensor(opt.batchSize,3,opt.targetSize,opt.targetSize)
    for i = 2, opt.batchSize do
      input[i] = boxesCPU[inpC][i] -- fill the input batch with other examples so that the batch normalization can work
    end
    
    for boxC = 1, #boxesCPU[inpC] do
      local box_wd = boxSizes[inpC][boxC][3]-boxSizes[inpC][boxC][1]+1;
      local box_ht = boxSizes[inpC][boxC][4]-boxSizes[inpC][boxC][2]+1;
      assert(box_wd == box_ht)
      input[1] = boxesCPU[inpC][boxC]
      local outputs = model:forward(input)
      local tmpOutput = outputs[1]
      tmpOutput = tmpOutput:float()

      local newOut = torch.Tensor(opt.numClasses,box_wd,box_ht)
      for k = 1,opt.numClasses do
        newOut[k] = image.scale(tmpOutput[k],box_wd,box_ht) 
      end

      tmp = padded_output[{{},{boxSizes[inpC][boxC][2],boxSizes[inpC][boxC][4]},{boxSizes[inpC][boxC][1],boxSizes[inpC][boxC][3]}}]
      padded_output[{{},{boxSizes[inpC][boxC][2],boxSizes[inpC][boxC][4]},{boxSizes[inpC][boxC][1],boxSizes[inpC][boxC][3]}}] = torch.cmax(tmp, newOut)
    end
    local x = inputsCPU[inpC]:size(3)
    local y = inputsCPU[inpC]:size(2)
    local final_score = padded_output[{{},{x,x+y-1},{y,x+y-1}}]
    
    local _,prediction_sorted = final_score:float():sort(1, true)
    prediction_sorted = prediction_sorted:float()

    final_out[inpC] = prediction_sorted[{1,{},{}}]
  end

  cutorch.synchronize()

  batchNumber = batchNumber + 1

  for i = 1,#inputsCPU do
	-- save results as mat files, so that they can be loaded in MATLAB and saved using the correct format and colormap for final testing on evaluation server
    matio.save(paths.concat(opt.save, 'finalImages/' .. imNames[i] .. '.mat'), final_out[i])    
  end

  print(('[%d/%d]\tTime(s) %.3f s \t Net: %s'):format(
      batchNumber, math.floor(numTestImages/opt.batchSize), timer:time().real, opt.netType))

  -- save some examples for the report.html
  if batchNumber == 1 then
    imgCount = #inputsCPU < 16 and #inputsCPU or 16
    for i = 1,imgCount do
      img = inputsCPU[i]:index(1,torch.LongTensor{3,2,1})
      img = img / 255
      for i=1,3 do -- channels
        if mean then img[{{i},{},{}}]:add(mean[i]) 
        else error('no mean given')
        end
      end
      saveImagesTest(img, final_out[i], i)    
    end
  end
end

