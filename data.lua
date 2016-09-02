local ffi = require 'ffi'
local threads = require 'threads'
threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey*.lua
-------------------------------------------------------------------------------
do -- start K datathreads (donkeys)
  if opt.nDonkeys > 0 then
    local options = opt -- make an upvalue to serialize over to donkey threads
    donkeys = threads.Threads(
      opt.nDonkeys,
      function()
        require 'torch'
      end,
      function(idx)
        opt = options -- pass to all donkeys via upvalue
        tid = idx
        local seed = opt.manualSeed + idx
        torch.manualSeed(seed)
        print(string.format('Starting donkey with id: %d seed: %d', tid, seed))
        paths.dofile('donkey' .. opt.dataset .. '.lua') --Stage1, Stage2, Pascal, BB, PascalTest
      end
    );
  else -- single threaded data loading. useful for debugging
    paths.dofile('donkey' .. opt.dataset .. '.lua') --Stage1, Stage2, Pascal, BB, PascalTest
    donkeys = {}
    function donkeys:addjob(f1, f2) f2(f1()) end
    function donkeys:synchronize() end
  end
end

-- get the total number of train/val/test images
if opt.testing then
  donkeys:addjob(function() return #testImages end, function(c) numTestImages = c end)
  donkeys:synchronize()
  assert(numTestImages, "Failed to get numTestImages")
  print('Number of testing images: ' .. numTestImages)
else
  donkeys:addjob(function() return #trainImages end, function(c) numTrainImages = c end)
  donkeys:synchronize()
  assert(numTrainImages, "Failed to get numTrainImages")
  print('Number of training images: ' .. numTrainImages)

  donkeys:addjob(function() return #valImages end, function(c) numValImages = c end)
  donkeys:synchronize()
  assert(numValImages, "Failed to get numValImages")
  print('Number of validation images: ' .. numValImages)
end

-- load the file that contains the mean (and std) values for the dataset
local meanStdFile = opt.meanFilePath
local meanstd = torch.load(meanStdFile)
mean = meanstd.mean
print('Loaded mean and std from cache')