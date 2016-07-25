require 'nn'
require 'cudnn'
require 'cunn'
require 'cutorch'
require 'xlua'
require 'optim'

---------------------------------------------------------------------------------
modelpath = '/data/sloewe/train/stackShare/2Stage/'
model = 'model_40.t7'
epochL = 2000
name = 'nn.SpatialBatchNormalization'
---------------------------------------------------------------------------------

local opts = paths.dofile('opts.lua')
opt = opts.parse(arg)

dofile 'data.lua'
dofile 'util.lua'

assert(cutorch.getDeviceCount() == opt.nGPU, 'Make GPUs invisible! - export CUDA_VISIBLE_DEVICES=0,1')

cudnn.benchmark = true
cudnn.fastest = true

torch.setnumthreads(opt.threads)
torch.manualSeed(opt.manualSeed)

cutorch.setDevice(opt.defGPU)

model = torch.load(modelpath .. model)

opt.batchSize = 8

function train()

--set model to test-mode, but leave BN-layers in trainmode to get mean/val values for individual batches
  model:evaluate()

  BNcount = 0
  BN_means = {}
  BN_vars = {}
  for k,v in pairs(model:findModules(name)) do
    v.train = true
 --   assert(v.momentum == 1)
    v.momentum = 1
    BNcount = k
    BN_means[k] = torch.zeros(v.running_mean:size())
    BN_vars[k] = torch.zeros(v.running_var:size())
  end

  local tic = torch.tic()

  local indices = torch.randperm(numTrainImages):long():split(opt.batchSize)
-- remove last element so that all the batches have equal size
  indices[#indices] = nil

  cutorch.synchronize()
  for t,v in ipairs(indices) do
    -- queue jobs to data-workers
    donkeys:addjob(
      -- the job callback (runs in data-worker thread)
      function() --load single batches
        local inputs, labels = loadTrainBatch(v)
        return inputs, labels
      end,
      -- the end callback (runs in the main thread)
      trainBatch
    )
    xlua.progress(t, epochL)
    if t == epochL then
      break
    end
  end

  donkeys:synchronize()
  cutorch.synchronize()

  for i = 1, #BN_means do
    BN_means[i] = BN_means[i] / epochL
    BN_vars[i] = BN_vars[i] / epochL
    BN_vars[i] = BN_vars[i] * (opt.batchSize / (opt.batchSize - 1))
  end
  
  --print(BN_means[1])
  --print(BN_vars[1])
  
  var_eps = 1e-9
  
  for k,v in pairs(model:findModules(name)) do
    gamma = v.weight:double()
    beta = v.bias:double()
    ex = BN_means[k]
    varx = BN_vars[k]
    
    new_gamma = torch.cdiv(gamma, torch.sqrt(varx + var_eps))
    new_beta = torch.add(beta, -(torch.cdiv(torch.cmul(gamma, ex), torch.sqrt(varx + var_eps))))
    
    v.weight = new_gamma:cuda()
    v.bias = new_beta:cuda()
    v.running_mean = torch.zeros(v.running_mean:size()):cuda()
    v.running_var = torch.add(torch.ones(v.running_var:size()),-v.eps):cuda()
  end
  
  model = model:cuda()
  model:evaluate()
  torch.save(modelpath .. 'modelBN.t7', model)
end

-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

meanLogger = optim.Logger(paths.concat(opt.save, 'mean.log'))
meanLogger:setNames{'mean'}
meanLogger.showPlot = false

varLogger = optim.Logger(paths.concat(opt.save, 'var.log'))
varLogger:setNames{'var'}
varLogger.showPlot = false


-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
  cutorch.synchronize()
  collectgarbage()
  timer:reset()

  -- transfer over to GPU
  inputs = inputs or (opt.nGPU == 1 and torch.CudaTensor() or cutorch.createCudaHostTensor())
  inputs:resize(inputsCPU:size()):copy(inputsCPU)

  cutorch.synchronize()
  local outputs = model:forward(inputs)
  cutorch.synchronize()


  for k,v in pairs(model:findModules(name)) do 
    BN_means[k] = BN_means[k] + v.running_mean:double()
    BN_vars[k] = BN_vars[k] + v.running_var:double()

    if k == 1 then
      meanLogger:add{v.running_mean[1]}
      meanLogger:style{'-'}
      meanLogger:plot()

      varLogger:add{v.running_var[1]}
      varLogger:style{'-'}
      varLogger:plot()
    end
  end

  dataTimer:reset()
end


train()
