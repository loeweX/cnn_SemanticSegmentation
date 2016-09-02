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
weights[{{1,opt.numClasses}}] = 1
weights[{{opt.numClasses+1}}] = 0 -- this label does not contribute to the error calculation

criterion = cudnn.SpatialCrossEntropyCriterion(weights)

print '==> here is the loss function:'
print(criterion)

----------------------------------------------------------------------
-- train on GPU
model:cuda()
model = makeDataParallel(model, opt.nGPU) -- defined in util.lua - training on multiple GPUs

criterion:cuda()