--[[
uses the batchnormalization directly after convolutions, before addition, but doesnt change original torch net structure
no shared weights in shortcut-connection
]]--

local Convolution = cudnn.SpatialConvolution  --LOCAL!
local Avg = cudnn.SpatialAveragePooling
local ReLU = cudnn.ReLU
local SBatchNorm = nn.SpatialBatchNormalization
local FullConvolution = cudnn.SpatialFullConvolution

local depth = 50 --opt.depth or 50
local shortcutType = 'B' --opt.shortcutType or 'B'
local layerNum = 8

-- The shortcut layer is either identity or 1x1 convolution
local function shortcut(nInputPlane, nOutputPlane, stride, adj)
  local useConv = shortcutType == 'C' or
  (shortcutType == 'B' and nInputPlane ~= nOutputPlane)
  if useConv then
    -- 1x1 convolution
    return nn.Sequential()
    :add(FullConvolution(nInputPlane, nOutputPlane, 1, 1, stride, stride, 0, 0, adj, adj))
    --:add(SBatchNorm(nOutputPlane))
 --[[ elseif nInputPlane ~= nOutputPlane then
    -- Strided, zero-padded identity shortcut
    return nn.Sequential()
    :add(nn.SpatialAveragePooling(1, 1, stride, stride))
    :add(nn.Concat(2)
      :add(nn.Identity())
      :add(nn.MulConstant(0))) ]]--
  else
    return nn.Identity()
  end
end

-- The bottleneck residual layer for 50, 101, and 152 layer networks
local function bottleneck(n, stride)
  local nInputPlane = n * 4
  if stride == 2 then
    nOutputPlane = n*2
    adj = 1
  else
    nOutputPlane = n*4
    adj = 0
  end  
  if nInputPlane == 256 and stride == 2 then
    nOutputPlane = 64
    adj = 0
    stride = 1
  end
  local s = nn.Sequential()
  s:add(FullConvolution(nInputPlane,n,1,1,1,1,0,0))
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  tmpLayer = FullConvolution(n,n,3,3,stride,stride,1,1,adj,adj)
  s:add(tmpLayer)
  s:add(SBatchNorm(n))
  s:add(ReLU(true))
  s:add(FullConvolution(n,nOutputPlane,1,1,1,1,0,0))
  --s:add(SBatchNorm(nOutputPlane))
  
  if layerNum >= 6 and stride == 2 then
    origLayer = model:get(layerNum):get(1):get(1):get(1):get(4)
    tmpLayer = tmpLayer:cuda()
    origLayer = origLayer:cuda()
    tmpLayer:share(origLayer,'weight','bias','gradWeight','gradBias')
    layerNum = layerNum - 1
  end
  
  return nn.Sequential()
  :add(nn.ConcatTable()
    :add(s)
    :add(shortcut(nInputPlane, nOutputPlane, stride, adj)))
  :add(nn.CAddTable(true))
  :add(SBatchNorm(nOutputPlane))
  :add(ReLU(true))
end

-- Creates count residual blocks with specified number of features
local function layer(block, features, count, stride)
  local s = nn.Sequential()
  for i=1,count do
    s:add(block(features, i == count and stride or 1))
  end  
  return s
end

-- Configurations for ResNet:
--  num. residual blocks, num features, residual block function
local cfg = {
  [50]  = {{3, 6, 4, 3}, 2048, bottleneck},
}

assert(cfg[depth], 'Invalid depth: ' .. tostring(depth))
local def, nFeatures, block = table.unpack(cfg[depth])
print(' | ResNet-' .. depth .. ' ImageNet')

model = torch.load('models/resnet-50.t7')

model:remove()
model:remove()
model:remove()

model:add(layer(block, 512, def[1], 2)) --fullconv 5
model:add(layer(block, 256, def[2], 2)) --fullconv 4
model:add(layer(block, 128, def[3], 2)) --fullconv 3
model:add(layer(block, 64, def[4], 2)) --fullconv 2

model:remove(4)
pool1 = nn.SpatialMaxPooling(2,2,2,2)
model:insert(pool1,4)

model:add(nn.SpatialMaxUnpooling(pool1))

model:get(12):get(3):remove(3)  -- remove last BatchNormalization

--model:add(Convolution(64, 21, 7, 7, 2, 2, 3, 3))
model:add(FullConvolution(64, 21, 7, 7, 2, 2, 3, 3, 1, 1))


local function ConvInit(name)
  for k,v in pairs(model:findModules(name)) do
   -- local n = v.kW*v.kH*v.nOutputPlane
    v.weight:normal(0,0.01)
    v.bias:zero()
--    v.gradBias:zero()
  end
end
local function BNInit(name)
  for k,v in pairs(model:findModules(name)) do
    v.weight:fill(1)
    v.bias:fill(0.001)
   -- v.momentum = 1
  end
end

--ConvInit('cudnn.SpatialConvolution')
ConvInit('cudnn.SpatialFullConvolution')
BNInit('nn.SpatialBatchNormalization')
