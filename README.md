# CXR_Embedding_Fairness , Underdiagnosis bias mitigation with expert foundation model's representation

All configuration files are located in the ./configs directory. Before running the training or testing commands, make sure to adjust the parameters in the .yaml files according to your requirements.

## Training
To train your model, use the following command. Replace ***.yaml with the name of your specific configuration file.

`python main.py fit -c ./configs/***_config.yaml `

## Testing
After training, you can test your model by specifying the path to the saved checkpoint and the configuration file used for training:

`python main.py test -c ./configs/***_config.yaml --ckpt_path ./path_to_saved_checkpoint `
