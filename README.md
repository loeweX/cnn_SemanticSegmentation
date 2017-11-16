# cnn_SemanticSegmentation

See BA_SindyLöwe_SemanticSegmentation.pdf for a description of the project.

## Dependencies:

1. Have a graphics card with Cuda 7.5 and cuDNN v5 (both from Nvidia) installed

2. Get [Torch](http://torch.ch/docs/getting-started.html)

3. Install an additional packages with these commands:

	`luarocks install  torch`    
	`luarocks install  image`    
	`luarocks install  paths`     
	`luarocks install  xlua`      
	`luarocks install  optim`    
	`luarocks install  nn`      
	`luarocks install  trepl`    
	`luarocks install  cunn`      
	`luarocks install  cudnn`      
	`luarocks install  cutorch`    
	`luarocks install  matio`   
	`luarocks install  mattorch`   


## How to use:

1. run:
		`th main.lua` 

2. run `th main.lua -h` to see available options.
