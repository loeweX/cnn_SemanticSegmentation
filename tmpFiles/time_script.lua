require 'nn'
require 'cudnn'
require 'cunn'

model = torch.load('model_40.t7')
model:cuda()

input = torch.CudaTensor(8,3,224,224)

t = 0
for i = 1,1000 do
  start = torch.tic()
  out = model:forward(input)
  t = t + torch.toc(start)
end
print(t/1000)


-- server:  0.031574970722198
-- pc:      0.10423856210709

--8batch
--DeconvNet - 