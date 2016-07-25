local M = { }

function M.parse(arg)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Fully Convolutional Net for Semantic Segmentation')
  cmd:text()
  cmd:text('Options:')
-- general stuff:
  cmd:option('-save', '/data/sloewe/train', 'subdirectory to save/log experiments in')  
  --cmd:option('-save', '/rahome/sloewe/torch/myProjects/results_fullconv_rgbd/important', 'subdirectory to save/log experiments in')
  cmd:option('-defGPU', 1, 'Default preferred GPU')
  cmd:option('-nGPU', 1, 'Number of GPUs to be used')

-- global:
  cmd:option('-manualSeed', 1, 'fixed input seed for repeatable experiments')
  cmd:option('-nDonkeys', 2, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-threads', 2, 'number of threads')
  cmd:option('-targetSize', 500, 'Size of the outputs of the net: targetSize x targetSize')
  cmd:option('-numClasses', 21, 'Number of Classes to be differentiated')
  cmd:option('-saveEpoch', 5, 'Number of epochs after which to save model and params') 
  cmd:option('-dataset', 'pascal', 'which dataset to train on: stage1 | stage2 | pascal')
  cmd:option('-trainValSplit', false, 'have a true split between val and train data?')

-- model:
  cmd:option('-netType', 'stackShare', 'net type to train: Deconv | FCN | Resnet | ...')
  cmd:option('-shareGradInput', false, 'Share gradInput tensors to reduce memory usage')
  
--retrain model
  --cmd:option('-retrain', 'none', 'provide path to model to retrain with')
  --cmd:option('-retrain', '/data/sloewe/results_deconv/Deconv/train8B/2Stage/model_40.t7', 'provide path to model to retrain with')
  cmd:option('-retrain', '/data/sloewe/train/stackShare/2Stage/model_40.t7', 'provide path to model to retrain with')
  cmd:option('-optimState', 'none', 'provide path to an optimState to reload from')
  cmd:option('-epochNumber', 1, 'Manual epoch number (useful on restarts)')

-- training:
  cmd:option('-numEpochs', 400, 'number of epochs to be trained')
  cmd:option('-epochLength', 1000, 'Number of Batches to train every epoch')
  cmd:option('-epochLengthVal', 100, 'Number of Batches to test every epoch')
  cmd:option('-epoch_step', 10000, 'number of epochs when to lower learningRate') 
  cmd:option('-batchSize', 2, 'mini-batch size')
  cmd:option('-fullBatchDiv', 1, 'train full batch (=1) or devided in x parts')
  cmd:option('-batchDifVal', 1, 'size reduction for validation')

-- optimization options
  cmd:option('-learningRate', 0.0001, 'learning rate at t=0') --start with 0.01
  cmd:option('-weightDecay', 0.0005, 'weight decay')
  cmd:option('-momentum', 0.9, 'momentum')
  
-- testing?
  cmd:option('-testing', false, 'testing final precision')
  cmd:option('-testFullSize', false, 'test on full images (true) or bounding boxes (false)')
  cmd:option('-numBoxes', 50, 'Number of bounding boxes to use for final testing')

  cmd:text()

  local opt = cmd:parse(arg or {})
  -- add commandline specified options
  opt.save = paths.concat(opt.save,
    cmd:string(opt.netType, opt,
      {retrain=true, optimState=true, cache=true, data=true}))
  -- add date/time
  opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))

  print('Will save at '..opt.save)
  paths.mkdir(opt.save)
  paths.mkdir(opt.save .. '/images')
  cmd:log(opt.save .. '/log', opt)

  return opt
end

return M
