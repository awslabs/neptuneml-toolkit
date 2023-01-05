Neptune ML Toolkit
------------------

The Neptune ML Toolkit is a python library for developing models for machine learning on graphs with `Amazon Neptune ML <https://aws.amazon.com/neptune/machine-learning/>`_, a feature of `Amazon Neptune <https://aws.amazon.com/neptune/>`_.
Amazon Neptune ML makes it possible to automatically train and deploy graph machine learning models to help find hidden patterns and extract useful insights from your heterogeneous graph data stored in the Amazon Neptune Graph Database.
Amazon Neptune ML uses Deep Graph Library `DGL <https://www.dgl.ai/>`_ and `Amazon SageMaker <https://aws.amazon.com/sagemaker/>`_ to build and train Graph Neural Networks (GNNs), a machine learning technique purpose-built for graphs, for tasks such as Node Property Prediction, Edge Property Prediction, Link Prediction.
See the `Neptune ML documentation <https://docs.aws.amazon.com/neptune/latest/userguide/machine-learning.html>`_ page to learn more.


With this toolkit, you can easily train and deploy GNN models with Neptune ML. You can interact with Neptune ML model management APIs with the `NeptuneMLClient` object.

You can also `develop custom models <https://docs.aws.amazon.com/neptune/latest/userguide/machine-learning-custom-models.html>`_ to be used with Amazon Neptune ML, using DGL and other python libraries.

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   neptuneml_toolkit/index.rst



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
