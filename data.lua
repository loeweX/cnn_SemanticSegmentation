local ffi = require 'ffi'
local threads = require 'threads'
threads.serialization('threads.sharedserialize')

-- This script contains the logic to create K threads for parallel data-loading.
-- For the data-loading details, look at donkey.lua
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

        if opt.testing then
          if opt.testFullSize then
            paths.dofile('donkeyPascalTest.lua')
          else
            paths.dofile('donkeyBB.lua')
          end  
        elseif opt.dataset == 'pascal' then
          paths.dofile('donkeyPascal.lua')
        elseif opt.dataset == 'stage1' then
          paths.dofile('donkeyStage1.lua')
        elseif opt.dataset == 'stage2' then
          paths.dofile('donkeyStage2.lua')
        end
      end
    );
  else -- single threaded data loading. useful for debugging
    if opt.testing then
      if opt.testFullSize then
        paths.dofile('donkeyPascalTest.lua')
      else
        paths.dofile('donkeyBB.lua')
      end
    elseif opt.dataset == 'pascal' then
      paths.dofile('donkeyPascal.lua')
    elseif opt.dataset == 'stage1' then
      paths.dofile('donkeyStage1.lua')
    elseif opt.dataset == 'stage2' then
      paths.dofile('donkeyStage2.lua')
    end
    donkeys = {}
    function donkeys:addjob(f1, f2) f2(f1()) end
    function donkeys:synchronize() end
  end
end



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