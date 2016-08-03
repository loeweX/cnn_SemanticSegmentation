--[[
how to use for training:
- adjust opts.lua to your needs
- write a donkey to load your data (look at donkeyTemplate for help)
]]--

require 'torch'
require 'paths'
require 'xlua'
require 'optim'
require 'nn'
require 'xlua'   
require 'image'

color = require 'trepl.colorize'
torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '==> processing options'

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

print('==> switching to CUDA')
require 'cunn'
require 'cudnn'
require 'cutorch'

assert(cutorch.getDeviceCount() == opt.nGPU, 'Make GPUs invisible! - export CUDA_VISIBLE_DEVICES=0,1')

cudnn.benchmark = true
cudnn.fastest = true

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.manualSeed)

cutorch.setDevice(1)

----------------------------------------------------------------------
print '==> executing all'

dofile 'util.lua'
dofile 'logger.lua'

dofile 'data.lua'
dofile 'model.lua'

----------------------------------------------------------------------

if opt.testing then
  if opt.testFullSize then
    dofile 'test.lua'
  else
    dofile 'testBB_fast.lua'
  end  
  test()
else
  dofile 'train.lua'
  dofile 'val.lua'
  epoch = 1

  tmp = torch.zeros(opt.numClasses)
  for i = 1,opt.numClasses do
    tmp[i] = 0
  end
  writeReport(tmp)

  print '==> training!'
  for i=1,opt.numEpochs do
    train()
    val()
    epoch = epoch+1
  end
end

