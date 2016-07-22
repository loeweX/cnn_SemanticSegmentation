print('Loading Model: ' .. opt.netType)


-- Create Network
-- If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
  assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
  print('Loading model from file: ' .. opt.retrain)
  model = torch.load(opt.retrain)
else
  paths.dofile('models/' .. opt.netType .. '.lua')
end

print('==> here is the model: ')
print(model)

-- define loss function
weights = torch.Tensor(opt.numClasses+1)
weights[{{1}}] = 1 --0.5
weights[{{2,opt.numClasses}}] = 1  --eventuell niedrigeres Gewicht für Hintergrund
weights[{{opt.numClasses+1}}] = 0 -- Randpixel nicht in Bewertung einbeziehen

criterion = cudnn.SpatialCrossEntropyCriterion(weights)

print '==> here is the loss function:'
print(criterion)

----------------------------------------------------------------------
-- train on GPU
model:cuda()

model = makeDataParallel(model, opt.nGPU) -- defined in util.lua - training on multiple GPUs

criterion:cuda()

----------------------------------------------------------------------

--[[ Caffe definition of the last layers (starting from first layer to be changed (originally Inner-Product Layer)

layers { bottom: 'pool5' top: 'fc6' name: 'fc6' type: CONVOLUTION
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { engine: CAFFE kernel_size: 7 num_output: 4096 } }
layers { bottom: 'fc6' top: 'fc6' name: 'relu6' type: RELU }
layers { bottom: 'fc6' top: 'fc6' name: 'drop6' type: DROPOUT
  dropout_param { dropout_ratio: 0.5 } }
layers { bottom: 'fc6' top: 'fc7' name: 'fc7' type: CONVOLUTION
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { engine: CAFFE kernel_size: 1 num_output: 4096 } }
layers { bottom: 'fc7' top: 'fc7' name: 'relu7' type: RELU }
layers { bottom: 'fc7' top: 'fc7' name: 'drop7' type: DROPOUT
  dropout_param { dropout_ratio: 0.5 } }
layers { name: 'score-fr' type: CONVOLUTION bottom: 'fc7' top: 'score'
  blobs_lr: 1 blobs_lr: 2 weight_decay: 1 weight_decay: 0
  convolution_param { engine: CAFFE num_output: 21 kernel_size: 1 } }

layers { type: DECONVOLUTION name: 'upsample' bottom: 'score' top: 'bigscore'
  blobs_lr: 0 blobs_lr: 0
  convolution_param { num_output: 21 kernel_size: 64 stride: 32 } }
layers { type: CROP name: 'crop' bottom: 'bigscore' bottom: 'data' top: 'upscore' }
]]--


if itorch then
  print '==> visualizing VGG filters'
  print('Layer 1 filters:')
  itorch.image(model:get(1).weight)
else
  print '==> To visualize filters, start the script in itorch notebook'
end

