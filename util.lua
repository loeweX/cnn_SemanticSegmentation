local ffi=require 'ffi'

function makeDataParallel(model, nGPU)
  if torch.type(model) == 'nn.DataParallelTable' then
    model = model:get(1)
  end

  -- This is useful for fitting a big net on 4 GPUs, but requires that all
  -- containers override backwards to call backwards recursively on submodules
  if opt.shareGradInput then
    local function sharingKey(m)
      local key = torch.type(m)
      if m.__shareGradInputKey then
        key = key .. ':' .. m.__shareGradInputKey
      end
      return key
    end

    -- Share gradInput for memory efficient backprop
    local cache = {}
    model:apply(function(m)
        local moduleType = torch.type(m)
        if torch.isTensor(m.gradInput) and moduleType ~= 'nn.ConcatTable' then
          local key = sharingKey(m)
          if cache[key] == nil then
            cache[key] = torch.CudaStorage(1)
          end
          m.gradInput = torch.CudaTensor(cache[key], 1, 0)
        end
      end)
    for i, m in ipairs(model:findModules('nn.ConcatTable')) do
      if cache[i % 2] == nil then
        cache[i % 2] = torch.CudaStorage(1)
      end
      m.gradInput = torch.CudaTensor(cache[i % 2], 1, 0)
    end
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

  cutorch.setDevice(opt.defGPU)

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