local M = { }

function M.parse(arg)

  local cmd = torch.CmdLine()
  cmd:text()
  cmd:text('Fully Convolutional Net for Semantic Segmentation')
  cmd:text()
  cmd:text('Options:')
-- general stuff:
  cmd:option('-save', '/data/sloewe/train', 'subdirectory to save/log experiments in')  
  cmd:option('-nGPU', 1, 'Number of GPUs to be used') -- make unused GPUs invisible to save memory 
                                                      -- export CUDA_VISIBLE_DEVICES=0,1

-- global:
  cmd:option('-manualSeed', 2, 'fixed input seed for repeatable experiments')
  cmd:option('-nDonkeys', 2, 'number of donkeys to initialize (data loading threads)')
  cmd:option('-threads', 2, 'number of threads for torch commands')
  cmd:option('-targetSize', 224, 'Size of the in-/outputs of the net: targetSize x targetSize') --other sizes "work" but dont give good results
  cmd:option('-numClasses', 21, 'Number of Classes to be distinguished')
  cmd:option('-saveEpoch', 5, 'Number of epochs after which to save model and params') 
  cmd:option('-dataset', 'Stage2', 'which dataset to train on: Stage1 | Stage2 | Pascal | BB | PascalTest | whateverNameYourDonkeyHas') 
  cmd:option('-trainValSplit', true, 'have a true split between val and train data?')
  
-- model:
  cmd:option('-netType', 'Deconv', 'net type to train: Deconv | FCN | Resnet | ...')
  
--retrain model
  --cmd:option('-retrain', 'none', 'provide path to model to retrain with')
  --cmd:option('-retrain', '/data/sloewe/results_deconv/Deconv/fullDataset8B/2Stage/model_40.t7', 'provide path to model to retrain with')
  cmd:option('-retrain', '/data/sloewe/train/Deconv/1StageSeed2/model_20.t7', 'provide path to model to retrain with')

-- training:
  cmd:option('-numEpochs', 400, 'number of epochs to be trained')
  cmd:option('-epochLength', 1000, 'Number of Batches to train every epoch')
  cmd:option('-epochLengthVal', 100, 'Number of Batches to test every epoch')
  cmd:option('-epoch_step', 10000, 'factor for learning rate reduction') 
  cmd:option('-batchSize', 64, 'mini-batch size')
  cmd:option('-fullBatchDiv', 8, 'train full batch (=1) or devided in x parts')
  cmd:option('-batchSizeVal', 8, 'batch size for validation (should be equal to batchSize / fullBatchDiv')

-- optimization options
  cmd:option('-learningRate', 0.01, 'learning rate at t=0') --start with 0.01
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
