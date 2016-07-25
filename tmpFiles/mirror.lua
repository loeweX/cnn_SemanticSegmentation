require 'nn'

one = nn.SpatialConvolution(3,64,1,1,2,2,0,0)
two = nn.SpatialFullConvolution(64,3,1,1,2,2,0,0,1,1)

model = nn.Sequential()
model:add(one)
model:add(two)


one.bias[{{1,-1}}]  = 0
two.bias[{{1,-1}}]  = 0

input = torch.zeros(1,3,224,224) + 0.1
output = model:forward(input)



two:share(one, 'weight','bias', 'gradWeight', 'gradBias')  --two gets the same weights as one

one = nn.SpatialConvolution(3,64,1,1,2,2,0,0)
two = nn.SpatialConvolution(3,64,1,1,2,2,0,0)
