local ffi=require 'ffi'

function makeDataParallel(model, nGPU)
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end

  -- Wrap the model with DataParallelTable, if using more than one GPU
  if opt.nGPU > 1 then
    local gpus = torch.range(1, opt.nGPU):totable()
    local fastest, benchmark = cudnn.fastest, cudnn.benchmark

    local dpt = nn.DataParallelTable(1, true, true)
    :add(model, gpus)
    :threads(function()
        local cudnn = require 'cudnn'
        cudnn.fastest, cudnn.benchmark = fastest, benchmark
      end)
    dpt.gradInput = nil

    model = dpt:cuda()
  end

  cutorch.setDevice(1)

  return model
end


local function dimnarrow(x,sz,pad,dim)
  local xn = x
  for i=1,x:dim() do
    if i > dim then
      xn = xn:narrow(i,pad[i]+1,sz[i])
    end
  end
  return xn
end

function padzero(x,pad)
  local sz = x:size()
  for i=1,x:dim() do sz[i] = sz[i]+pad[i]*2 end
  local xx = x.new(sz):zero()
  local xn = dimnarrow(xx,x:size(),pad,-1)
  xn:copy(x)
  return xx
end