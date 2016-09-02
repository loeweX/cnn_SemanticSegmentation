--[[
Faster reimplementation of testBB.lua that sends several bounding box instances through the network at once.
Should work, but final results were not tested yet
]]--

require 'xlua'
local matio = require 'matio'

print '==> defining testing procedure'

local meanStdFile = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation/meanStd.t7'
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
  -- remove last element so that all the batches have equal size
  --indices[#indices] = nil

  -- assert(numTestImages % opt.batchSize == 0, 'Batchsize must fit overall test image number')

  print('testing: ')

  epochL = #indices

  cutorch.synchronize()
  for t,v in ipairs(indices) do
    -- queue jobs to data-workers
    donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function() --load single batches
        inputs, imNames, boxImgs, boxSizes = loadTestBatch(v)
        return inputs, imNames, boxImgs, boxSizes
      end,
      -- the end callback (runs in the main thread)
      testBatch
    )
    -- xlua.progress(t, epochL)
  end

  donkeys:synchronize()
  cutorch.synchronize()

  print(string.format('[TESTING SUMMARY] Total Time(s): %.2f', torch.toc(tic)))
  print('\n')

  writeReportTest()

  collectgarbage()
end


-- GPU inputs (preallocate)
local boxes = torch.CudaTensor()

function testBatch(inputsCPU, imNames, boxesCPU, boxSizes)
  cutorch.synchronize()
  collectgarbage()
  -- transfer over to GPU
  timer:reset()
  final_out = {}
  local input = torch.CudaTensor(opt.batchSize,3,224,224)

  for inpC = 1, #inputsCPU do
    local padded_output = torch.zeros(opt.numClasses,inputsCPU[inpC]:size(2) + 2 * inputsCPU[inpC]:size(3),inputsCPU[inpC]:size(3) + 2 * inputsCPU[inpC]:size(2))
    padded_output[1] = 0.00001


    for boxC = 1, #boxesCPU[inpC], opt.batchSize do
      local box_wd = torch.zeros(opt.batchSize)
      local box_ht = torch.zeros(opt.batchSize);
      for i = 0, opt.batchSize-1 do
        if (boxC+i) < #boxesCPU then
          box_wd[i+1] = boxSizes[inpC][boxC+i][3]-boxSizes[inpC][boxC+i][1]+1;
          box_ht[i+1] = boxSizes[inpC][boxC+i][4]-boxSizes[inpC][boxC+i][2]+1;
          assert(box_wd[i+1] == box_ht[i+1])
          input[i+1] = boxesCPU[inpC][boxC+i]
        end
      end

      local outputs = model:forward(input)

      for i = 0, opt.batchSize-1 do
        if (boxC+i) < #boxesCPU then
          local tmpOutput = outputs[i+1]
          tmpOutput = tmpOutput:float()

          local newOut = torch.Tensor(opt.numClasses,box_wd[i+1],box_ht[i+1])
          for k = 1,opt.numClasses do
            newOut[k] = image.scale(tmpOutput[k],box_wd[i+1],box_ht[i+1]) 
          end

          tmp = padded_output[{{},{boxSizes[inpC][boxC+i][2],boxSizes[inpC][boxC+i][4]},{boxSizes[inpC][boxC+i][1],boxSizes[inpC][boxC+i][3]}}]
          padded_output[{{},{boxSizes[inpC][boxC+i][2],boxSizes[inpC][boxC+i][4]},{boxSizes[inpC][boxC+i][1],boxSizes[inpC][boxC+i][3]}}] = torch.cmax(tmp, newOut)
        end
      end
    end
    local x = inputsCPU[inpC]:size(3)
    local y = inputsCPU[inpC]:size(2)
    local final_score = padded_output[{{},{x,x+y-1},{y,x+y-1}}]

    --get full size result, take only values > 0 , and multiply elementwise with this result
    --[[
    zero_mask = zeros(size(fcn_score));
    fcn_score = max(zero_mask,fcn_score);
        
    ens_score = deconv_score .* fcn_score;
    [ens_segscore, ens_segmask] = max(ens_score, [], 3); -- hier: 1.dimension!
    ens_segmask = uint8(ens_segmask-1); ]]--

    local _,prediction_sorted = final_score:float():sort(1, true)
    prediction_sorted = prediction_sorted:float()

    final_out[inpC] = prediction_sorted[{1,{},{}}]
  end

  cutorch.synchronize()

  batchNumber = batchNumber + 1

  for i = 1,#inputsCPU do
    matio.save(paths.concat(opt.save, 'finalImages/' .. imNames[i] .. '.mat'), final_out[i])   
  end

  print(('[%d/%d]\tTime(s) %.3f s \t Net: %s'):format(
      batchNumber, math.floor(numTestImages/opt.batchSize), timer:time().real, opt.netType))

  if batchNumber == 1 then
    imgCount = #inputsCPU < 16 and #inputsCPU or 16
    for i = 1,imgCount do
      img = inputsCPU[i]:index(1,torch.LongTensor{3,2,1})
      img = img / 255
      for i=1,3 do -- channels
        --     if std then img[{{i},{},{}}]:mul(std[i]) 
        --     else error('no std given')
        --     end
        if mean then img[{{i},{},{}}]:add(mean[i]) 
        else error('no mean given')
        end
      end
      saveImagesTest(img, final_out[i], i)    
    end
  end
end

