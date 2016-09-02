# cnn_SemanticSegmentation
============

## Dependencies:

1. Have a graphics card with Cuda 7.5 and cuDNN v5 (both from Nvidia) installed

2. Get [Torch](http://torch.ch/docs/getting-started.html)

3. Install an additional packages with these command:

    `luarocks install  image`
	`luarocks install  torch`
	`luarocks install  paths`
	`luarocks install  xlua`
	`luarocks install  optim`
	`luarocks install  nn`
	`luarocks install  xlua`   
	`luarocks install  trepl`   
	`luarocks install  cunn`   
	`luarocks install  cudnn`   
	`luarocks install  cutorch`   
	`luarocks install  matio`   
	`luarocks install  mattorch`   

4. Download the network myModel.t7 and meanStd.t7 from: 

    ???


## How to use:

1. adjust parameters on opts.lua to your needs

2. run:
		`th main.lua` 
