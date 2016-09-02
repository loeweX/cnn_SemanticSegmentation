-- test procedure that feeds the full images to the network at once

local matio = require 'matio'

print '==> defining testing procedure'

local batchNumber = 0
paths.mkdir(opt.save .. '/finalImages')

imagePath = opt.testImagesPath

local meanStdFile = opt.meanFilePath
local meanstd = torch.load(meanStdFile)
mean = meanstd.mean
std = meanstd.std
print('Loaded mean and std from cache')

function test()
  model:evaluate()  

  batchNumber = 0
  local tic = torch.tic()

  indices = torch.randperm(numTestImages):long():split(opt.batchSize)

  assert(numTestImages % opt.batchSize == 0, 'Batchsize must fit overall test image number')

  print('testing: ')

  epochL = #indices

  cutorch.synchronize()
  for t,v in ipairs(indices) do
    donkeys:addjob(
      function() --load single batches
        local inputs, imNames = loadTestBatch(v)
        return inputs, imNames
      end,
      testBatch
    )
    xlua.progress(t, epochL)
  end

  donkeys:synchronize()
  cutorch.synchronize()

  print(string.format('[TESTING SUMMARY] Total Time(s): %.2f', torch.toc(tic)))
  print('\n')

  writeReportTest()

  collectgarbage()
end


-- preallocate GPU inputs
local inputs = torch.CudaTensor()

function testBatch(inputsCPU, imNames)
  cutorch.synchronize()
  collectgarbage()

  -- transfer over to GPU
  inputs = inputs or (opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
  inputs:resize(inputsCPU:size()):copy(inputsCPU)

  -- feed inputs to netwotk
  local outputs = model:forward(inputs)

  cutorch.synchronize()

  batchNumber = batchNumber + 1

  _,prediction_sorted = outputs:float():sort(2, true):float()

  for i = 1,opt.batchSize do
    output = prediction_sorted[{i,1,{},{}}]
    img = image.load(imagePath .. '/' .. imNames[i] .. '.jpg') 

	-- transform output back to original image size
    tmpOut = image.scale(output, 500,500, 'simple')    
    finalOut = tmpOut[{{torch.floor((500-img:size(2))/2) + 1, 500 - torch.ceil((500-img:size(2))/2)}, {torch.floor((500-img:size(3))/2) + 1, 500 - torch.ceil((500-img:size(3))/2)}}]

    matio.save(paths.concat(opt.save, 'finalImages/' .. imNames[i] .. '.mat'), finalOut)   
  end

  -- save some examples for the report.html
  if batchNumber == 1 then
    imgCount = opt.batchSize < 16 and opt.batchSize or 16
    for i = 1,imgCount do
      img = inputsCPU[i]:index(1,torch.LongTensor{3,2,1})
      img = img/255
      for i=1,3 do -- channels
        if mean then img[{{i},{},{}}]:add(mean[i]) 
        else error('no mean given')
        end
      end
      saveImagesTest(img, prediction_sorted[{i,1,{},{}}], i)    
    end
  end
end