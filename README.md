**Shape prediction**

This is a shape prediction project that classifies the shape of nanoobject based on the intensity variation.

The software makes it possible to:
- train/improve the predictions models
- use trained model to perform predictions

There is a build-in help functionality.

**Usage**

***Training***

The app comes pre-trained on the included data. Training is only needed if new data is added.

The app responsible for training is _train.py_.

The data for training is provided in the cvs format and is stored in the data directory. Training data for a given shape is provided in a file with a name corresponding to that shape. In order to add the capability of recognizing a new shape, just add a properly named file with shape examples, the system will recognize this as a new shape.

To see the available training options please consult the current help (_python train.py --help_). It will always provide the most recent options. There are some notable features to know of:
- the command line switches can be combined, you may e.g. choose an arbitrary collection of prediction algorithms to be included
- if you select stacked switch then a variant with all algorithms stacked will be added
- you may decide to run PCA on input variables. PCA can be wither specified as a number of components (values from 1 up) or percentage of explained variance (0..1)
- you may decided to train a model on a continous subset of columns from the training set (for example just columns from 100 to 300) 
- the trained models are stored in the models directory. They can be identified by specifying a _tag_. If you do not specify a tag, a default model will be replaced.
- there is an integration with Neptune.ai provided (upload option). If you wish to use it, specify _NEPTUNE_API_TOKEN_ and _NEPTUNE_PROJECT_NAME_ in the shell  

***Predictions***

There are 3 ways to access the prediction models:
- command line - full access to all prediction models
- web interface - only default model available

The command line access is provided by _predict.py_. All available models are accessible this way. Please consult help (_python predict.py--help_) to find out the usage syntax.

The web interface is based on the Streamlit integration and requires streamlit. To access it locally please run '_streamlit run onlinepredict.py_'

**Notes**

SessionState used in the webinterface is based on the following gist: https://gist.github.com/tvst/036da038ab3e999a64497f42de966a92
