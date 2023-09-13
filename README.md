### File structure
'''
├── data
│   ├── training_images
│   ├── dev_images
│   ├── vlsp2023_dev_data.json
│   ├── vlsp2023_train_data.json
├── src
│   ├── dataset
│   │   ├── components
│   ├── views
│   ├── model
│   │   ├── components
│   │   │   ├── language
│   │   │   │   ├── encoders.py
│   │   │   ├── vision
│   │   │   │   ├── encoders.py
│   │   ├── model.py
│   ├── utils
├── README.md
├── requirements.txt
└── .gitignore
'''
### ToDo
- encoders.py contains layers/custom layers etc
- model.py contains final complete net (all language/vision encoders and final vision-language decoders/auxilary modules)
- dataset should contain custom dataset object/dataloader and datamodule if using lightning
