## Custom Model Zoo RGCN for Neptune ML Node Regression

This folder contains the source code implementation for how to use the RGCN model in the toolkit model zoo to implement a custom model that is trainable with Neptune ML.

To test run the model implementation locally, first make sure the neptune ml toolkit is already installed.

Then, download the processed graph data from the S3 output location of a [Neptune ML data processing job](https://docs.aws.amazon.com/neptune/latest/userguide/machine-learning-on-graphs-processing.html#machine-learning-on-graphs-processing-managing) for the node regression task. E.g:

* `aws s3 cp --recursive <s3://(S3 bucket name)/(path to data processing output)> data/`

The contents of this folder should contain at least: `graph.bin` and `info.pkl`.

Next create a folder to store the training output

* `mkdir out/`

Then run the following command to run the modeltraining step locally
* `python src/train.py --local `

You can examine the `train.py` to see the hyperparameters supported and the defaults that have been set for them.
You can modify these hyperparameters by changing the defaults or passing new values as command line arguments to `src/train.py`.

Once training is complete, you can run the command below to run the modeltransform step locally as well

* `python src/transform.py --local`

### Modifying the implementation

If you're adapting this example for node regression for a different dataset or with another model implementation, you need to modify `model-hpo-configuration.json` to account for your new model hyperparameters as well as the command line arguments in `train.py`.
However, you must keep following arguments as they always passed to your training script by the Neptune ML training infrastructure:
* `--name`
* `--model`
* `--task`
* `--target_ntype`
* `--property`
