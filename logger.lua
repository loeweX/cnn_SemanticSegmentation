require 'optim'   -- an optimization package, for online and batch methods
require 'image'

--[[  Write a html-file with all relevant logging information
To use add something like this to your script: 

  val_acc = acc_epoch
  val_loss = loss_epoch
  
  accLogger:add{train_acc, val_acc}
  accLogger:style{'-','-'}
  accLogger:plot()

  lossLogger:add{train_loss, val_loss}
  lossLogger:style{'-','-'}
  lossLogger:plot()

  writeReport()
  
This updates the current accuracy and loss.
the Log also saves input and output images of the net with their corresponding label, you have to save them in your program to show them in the report
To save Images do:

  if batchNumber == 1 then
    correct = torch.zeros(opt.batchSize, opt.targetSize, opt.targetSize) + opt.numClasses + 1
    correct = correct:float()
    _,prediction_sorted = outputs:float():sort(2, true) -- descending
    prediction_sorted = prediction_sorted:float()
    labelCopy = labels:float()
    locsForCorrect = labelCopy:eq(prediction_sorted[{{},1,{},{}}])
    correct:maskedFill(locsForCorrect,1)--labelCopy)
    
    imgCount = opt.batchSize < 16 and opt.batchSize or 16
    for i = 1,imgCount do
      saveImages(inputs[i], prediction_sorted[{i,1,{},{}}], labels[i], correct[i], 'train', i)    
    end
    if epoch % 2 == 0 then
      tmpStr = 'trEpoch' .. epoch
      saveImages(inputs[1], prediction_sorted[{1,1,{},{}}], labels[1], correct[1], tmpStr, 1)  
    end
  end
  
]]--

accLogger = optim.Logger(paths.concat(opt.save, 'acc.log'))
accLogger:setNames{'% mean accuracy (train set)', '% mean accuracy (val set)'}
accLogger.showPlot = false

lossLogger = optim.Logger(paths.concat(opt.save, 'loss.log'))
lossLogger:setNames{'% mean loss (train set)', '% mean loss (val set)'}
lossLogger.showPlot = false

train_acc = 0 
train_loss = 0
val_acc = 0
val_loss = 0

local acc_dir = paths.concat(opt.save, 'acc.png')
local loss_dir = paths.concat(opt.save, 'loss.png')      

classes = { '0=background' , '1=aeroplane', '2=bicycle', '3=bird', '4=boat', '5=bottle', '6=bus','7=car', '8=cat', '9=chair', '10=cow', '11=diningtable', '12=dog', '13=horse',   '14=motorbike', '15=person', '16=potted plant', '17=sheep', '18=sofa', '19=train',        
  '20=tv/monitor'} 

function writeReport(valAccuracy)

  os.execute(('convert -density 200 %s/acc.log.eps %s/acc.png'):format(opt.save,opt.save))
  os.execute(('convert -density 200 %s/loss.log.eps %s/loss.png'):format(opt.save,opt.save))

  local file = io.open(opt.save..'/report.html','w')
  file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <h4>Log: %s</h4>
      <h4>Epoch: %s</h4>
      <h4> Accuracy: </h4>
      <img src=%s> </br>
      <h4> Loss: </h4>
      <img src=%s>
      <h4> Accuracies in Val-Set:</h4>
      <table>
      ]]):format(opt.save,epoch,acc_dir,loss_dir))      

  for k,v in pairs(classes) do
    file:write('<tr> <td>'..v..'</td> <td>'..valAccuracy[k] ..'</td> </tr> \n')
  end

  mean_acc = valAccuracy:mean()
  file:write('<tr> <td> MEAN </td> <td>'.. valAccuracy:mean() ..'</td> </tr> \n')

-----------------------------------------------------------------------------------------

  file:write([[</table>
  <h4> OptimState: </h4>
  <table>
  ]])

  for k,v in pairs(optimState) do
    if torch.type(v) == 'number' then
      file:write('<tr> <td>'..k..'</td> <td>'..v..'</td> </tr> \n')
    end
  end

-----------------------------------------------------------------------------------------

  file:write([[</table>
  <h4> Opts: </h4>
  <table>
  ]])

  for k,v in pairs(opt) do
    if torch.type(v) == 'number' or torch.type(v) == string then
      file:write('<tr> <td>'..k..'</td> <td>'..v..'</td> </tr> \n')
    end
  end

-----------------------------------------------------------------------------------------

  file:write([[
    </table>
    <h4> Train Images </h4>
    input image  - output image - correct label - correct predicted </br>
    ]])

--input and output images
  imgCount = opt.batchSize < 16 and opt.batchSize or 16
  for i = 1, imgCount do
    input_dir = paths.concat(opt.save, 'images/trInputImg' .. i .. '.png')      
    output_dir = paths.concat(opt.save, 'images/trOutputImg' .. i .. '.png')      
    label_dir = paths.concat(opt.save, 'images/trLabelImg' .. i .. '.png') 
    correct_dir = paths.concat(opt.save, 'images/trCorrectImg' .. i .. '.png') 

    file:write(([[
      <h5> ImagePair: %s </h5>    
      <img src=%s>
      <img src=%s>
      <img src=%s>
      <img src=%s> </br>
      ]]):format(i, input_dir, output_dir, label_dir, correct_dir))     
  end

-----------------------------------------------------------------------------------------

  file:write([[
    </table>
    <h4> Val Images </h4>
    input image  - output image - correct label - correct predicted </br>
    ]])

--input and output images
  imgCount = opt.batchSize < 16 and opt.batchSize or 16
  for i = 1, imgCount do
    input_dir = paths.concat(opt.save, 'images/inputImg' .. i .. '.png')      
    output_dir = paths.concat(opt.save, 'images/outputImg' .. i .. '.png')      
    label_dir = paths.concat(opt.save, 'images/labelImg' .. i .. '.png') 
    correct_dir = paths.concat(opt.save, 'images/correctImg' .. i .. '.png') 

    file:write(([[
      <h5> ImagePair: %s </h5>    
      <img src=%s>
      <img src=%s>
      <img src=%s>
      <img src=%s> </br>
      ]]):format(i, input_dir, output_dir, label_dir, correct_dir))     
  end

-----------------------------------------------------------------------------------------

  file:write([[
    </table>
    <h4> Train Images over Epochs </h4>
    input image  - output image - correct label - correct predicted </br>
    ]])

--input and output images
  for i = 1, epoch do
    tmpStr = 'trEpoch' .. i
    input_dir = paths.concat(opt.save, 'images/inputImg' .. tmpStr .. '.png')      
    output_dir = paths.concat(opt.save, 'images/outputImg' .. tmpStr .. '.png')      
    label_dir = paths.concat(opt.save, 'images/labelImg' .. tmpStr .. '.png') 
    correct_dir = paths.concat(opt.save, 'images/correctImg' .. tmpStr .. '.png') 

    file:write(([[
      <h5> Epoch: %s </h5>    
      <img src=%s>
      <img src=%s>
      <img src=%s>
      <img src=%s> </br>
      ]]):format(i, input_dir, output_dir, label_dir, correct_dir))     
  end

-----------------------------------------------------------------------------------------

  file:write([[
    </table>
    <h4> Val Images over Epochs </h4>
    input image  - output image - correct label - correct predicted </br>
    ]])

--input and output images
  for i = 1, epoch do
    tmpStr = 'teEpoch' .. i
    input_dir = paths.concat(opt.save, 'images/inputImg' .. tmpStr .. '.png')      
    output_dir = paths.concat(opt.save, 'images/outputImg' .. tmpStr .. '.png')      
    label_dir = paths.concat(opt.save, 'images/labelImg' .. tmpStr .. '.png') 
    correct_dir = paths.concat(opt.save, 'images/correctImg' .. tmpStr .. '.png') 

    file:write(([[
      <h5> Epoch: %s </h5>    
      <img src=%s>
      <img src=%s>
      <img src=%s>
      <img src=%s> </br>
      ]]):format(i, input_dir, output_dir, label_dir, correct_dir))     
  end

-----------------------------------------------------------------------------------------

  file:write([[
    <h4> Model: </h4>
    <pre> ]])
  file:write(tostring(model))

  file:write'</pre></body></html>'
  file:close()

end


function saveImages(input, output, label, correct, trainVal, count)
  inputImg = input
  inputImg = inputImg:float()

  outputImg = output
  outputImg = outputImg:float()
  outputImg = imageToJet(outputImg)  

  labelsImg = label
  labelsImg = labelsImg:float()
  labelsImg = imageToJet(labelsImg)

  correctImg = correct
  correctImg = correctImg:float()
  correctImg = imageToJet(correctImg)

  if trainVal == 'train' then
    image.save(paths.concat(opt.save, 'images/trInputImg' .. count .. '.png'), inputImg)
    image.save(paths.concat(opt.save, 'images/trOutputImg' .. count .. '.png'), outputImg) 
    image.save(paths.concat(opt.save, 'images/trLabelImg' .. count .. '.png'), labelsImg)
    image.save(paths.concat(opt.save, 'images/trCorrectImg' .. count .. '.png'), correctImg)
  elseif trainVal == 'val' then
    image.save(paths.concat(opt.save, 'images/inputImg' .. count .. '.png'), inputImg)
    image.save(paths.concat(opt.save, 'images/outputImg' .. count .. '.png'), outputImg)      
    image.save(paths.concat(opt.save, 'images/labelImg' .. count .. '.png'), labelsImg)
    image.save(paths.concat(opt.save, 'images/correctImg' .. count .. '.png'), correctImg)
  else
    image.save(paths.concat(opt.save, 'images/inputImg' .. trainVal .. '.png'), inputImg)
    image.save(paths.concat(opt.save, 'images/outputImg' .. trainVal .. '.png'), outputImg)   
    image.save(paths.concat(opt.save, 'images/labelImg' .. trainVal .. '.png'), labelsImg)
    image.save(paths.concat(opt.save, 'images/correctImg' .. trainVal .. '.png'), correctImg)
  end
end

myColormap = image.colormap(opt.numClasses+1)
myColormap[1] = torch.zeros(3,1)
myColormap[opt.numClasses+1] = torch.ones(3,1)

function imageToJet(img)
  tmpImg = torch.zeros(3,img:size(1),img:size(2))--opt.targetSize, opt.targetSize)

  for i = 1,opt.numClasses+1 do
    locsTmp = img:eq(i)
    for j = 1,3 do
      tmpImg[j]:maskedFill(locsTmp,myColormap[{i,j}])
    end
  end
  return tmpImg
end

----------------------------------------------------------------------------------------------------------------------
-- TESTING PROCEDURES
function saveImagesTest(input, output, count)
  input = input:float()

  output = output:float()
  --output = imageToJet(output)  

  image.save(paths.concat(opt.save, 'images/inputImg' .. count .. '.png'), input)
  torch.save(paths.concat(opt.save, 'images/outputImgT' .. count .. '.t7'), output) 
    
  output = imageToJet(output)  
  image.save(paths.concat(opt.save, 'images/outputImg' .. count .. '.png'), output)   
end


function writeReportTest()

  local file = io.open(opt.save..'/report.html','w')
  file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <h4>Log: %s</h4>
      <h4> Test Images </h4>
      input image  - output image </br>
      ]]):format(opt.save))      

--input and output images
  imgCount = opt.batchSize < 16 and opt.batchSize or 16
  for i = 1, imgCount do
    input_dir = paths.concat(opt.save, 'images/inputImg' .. i .. '.png')      
    output_dir = paths.concat(opt.save, 'images/outputImg' .. i .. '.png')      

    file:write(([[
      <h5> ImagePair: %s </h5>    
      <img src=%s>
      <img src=%s> </br>
      ]]):format(i, input_dir, output_dir))     
  end
end

--------------------------------------------------------------
function writeFinalReportTest()
--  dataPath = '/data/DNN-common/Pascal2012/VOCdevkit/VOC2012/ImageSets/Segmentation'
  dataPath = '/data/DNN-common/DeconvPascal2012/VOC2012_TEST/ImageSets/Segmentation'
  local testImagesFile = paths.concat(dataPath, 'testImages.t7')
  testLoader = torch.load(testImagesFile)
  testImages = testLoader.testImages
  file = io.open('report.html','w')
  file:write(([[
      <!DOCTYPE html>
      <html>
      <body>
      <h4> Test Images </h4>
      input image  - output image </br>
      ]]):format())      
--input and output images
  for i = 1, #testImages do
    name = testImages[i]
    input_dir = paths.concat('./results/important/Deconv/testImages/' .. name .. '.jpg')      
    output_dir = paths.concat('./results/important/Deconv/matlabResults/' .. name .. '.png')      
    file:write(([[
      <h5> ImagePair: %s </h5>    
      <img src=%s>
      <img src=%s> </br>
      ]]):format(i, input_dir, output_dir))     
  end
end
