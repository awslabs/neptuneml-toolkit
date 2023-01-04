###########################
Neptune ML Toolkit
###########################
The Neptune ML Toolkit is a python library for developing models for machine learning on graphs with [Amazon Neptune ML](https://aws.amazon.com/neptune/machine-learning/), a feature of [Amazon Neptune](https://aws.amazon.com/neptune/).
Amazon Neptune ML makes it possible to automatically train and deploy graph machine learning models to help find hidden patterns and extract useful insights from your heterogeneous graph data stored in the Amazon Neptune Graph Database.
Amazon Neptune ML uses Deep Graph Library ([DGL](https://www.dgl.ai/)) and [Amazon SageMaker](https://aws.amazon.com/sagemaker/) to build and train Graph Neural Networks (GNNs), a machine learning technique purpose-built for graphs, for tasks such as Node Property Prediction, Edge Property Prediction, Link Prediction.
See the [Neptune ML documentation](https://docs.aws.amazon.com/neptune/latest/userguide/machine-learning.html) page to learn more.


With this toolkit, you can easily train and deploy GNN models with Neptune ML. You can interact with Neptune ML model management APIs with the `NeptuneMLClient` object.
You can also [develop custom models](https://docs.aws.amazon.com/neptune/latest/userguide/machine-learning-custom-models.html) to be used with Amazon Neptune ML, using DGL and other python libraries.
The toolkit contains [examples](./examples/custom-models) of custom model implementations and a model zoo with DGL implementations for popular heterogeneous graph models.


API Reference
=============

This API Reference details the usage of functions and objects included in the Neptune ML toolkit.

.. toctree::
   :maxdepth: 2
   :hidden:

   neptuneml_toolkit