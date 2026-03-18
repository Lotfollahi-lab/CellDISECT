=========
Tutorials
=========

.. raw:: html

    <div class="tutorials-header">
        <p class="tutorials-intro">
            Welcome to the CellDISECT tutorials section. Here you'll find comprehensive guides and practical examples 
            that will help you master CellDISECT's capabilities for single-cell analysis and counterfactual predictions.
        </p>
    </div>

Getting Started
-------------
Before diving into the tutorials, make sure you have:

* :doc:`Installed CellDISECT </installation>`
* Familiarized yourself with the :doc:`basic concepts </api/index>`
* Prepared your single-cell data in the appropriate format

Tutorial Categories
-----------------

Beginner Tutorials
~~~~~~~~~~~~~~~~

.. grid:: 2

    .. grid-item-card:: 🔬 Basic CellDISECT Training
        :link: CellDISECT_Counterfactual.ipynb
        :class-card: tutorial-card

        Learn how to train CellDISECT and make counterfactual predictions using the Kang dataset.
        Perfect for first-time users!

        +++
        :doc:`Open Notebook <CellDISECT_Counterfactual.ipynb>`

Perturbation Prediction
~~~~~~~~~~~~~~~~~~~~

.. grid:: 2

    .. grid-item-card:: 🧪 Perturbation Prediction
        :link: CellDISECT_Perturbation.ipynb
        :class-card: tutorial-card

        Predict gene expression under seen, unseen, and combinatorial perturbations
        using predefined gene embeddings (GenePT, ESM, scGPT).

        +++
        :doc:`Open Notebook <CellDISECT_Perturbation.ipynb>`

Advanced Applications
~~~~~~~~~~~~~~~~~~

.. grid:: 2

    .. grid-item-card:: 🧬 Latent Space Exploration
        :link: Erythroid_subset_inference.ipynb
        :class-card: tutorial-card

        Explore how to combine CellDISECT latent spaces for erythroid subset inference, 
        demonstrating advanced usage with Z_0 + Z_Organ integration.

        +++
        :doc:`Open Notebook <Erythroid_subset_inference.ipynb>`

    .. grid-item-card:: 🔄 Double Counterfactual Analysis
        :link: Eraslan_CF_Tutorial.ipynb
        :class-card: tutorial-card

        Advanced tutorial recreating Scenario 2 counterfactual predictions on the Eraslan dataset,
        as featured in our paper.

        +++
        :doc:`Open Notebook <Eraslan_CF_Tutorial.ipynb>`

.. note::
    Each tutorial includes downloadable Jupyter notebooks that you can run locally. 
    The notebooks are extensively documented with step-by-step explanations and best practices.

Detailed Tutorial Contents
------------------------

.. toctree::
    :maxdepth: 2
    :caption: Available Tutorials
    :numbered:

    CellDISECT_Counterfactual.ipynb
    CellDISECT_Perturbation.ipynb
    Erythroid_subset_inference.ipynb
    Eraslan_CF_Tutorial.ipynb

Tutorial Details
--------------

Basic Training Tutorial
~~~~~~~~~~~~~~~~~~~~~
In this introductory tutorial, you'll learn:

* How to prepare your data for CellDISECT
* Basic model training and configuration
* Making simple counterfactual predictions
* Visualizing and interpreting results

:download:`Download Notebook <CellDISECT_Counterfactual.ipynb>`

Perturbation Prediction Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This tutorial covers:

* Preparing predefined gene embeddings (GenePT, ESM) in ``adata.uns``
* Setting up ``setup_anndata`` with ``perturbation_key`` and ``perturbation_embedding_key``
* Training CellDISECT with perturbation-aware embeddings
* Predicting seen, unseen, and combinatorial perturbations
* Evaluating predictions with ``perturbation_metrics``

:download:`Download Notebook <CellDISECT_Perturbation.ipynb>`

Latent Space Analysis
~~~~~~~~~~~~~~~~~~~
This advanced tutorial covers:

* Understanding CellDISECT's latent space structure
* Combining multiple latent spaces (Z_0 + Z_Organ)
* Advanced visualization techniques
* Interpreting latent space representations

:download:`Download Notebook <Erythroid_subset_inference.ipynb>`

Double Counterfactual Tutorial
~~~~~~~~~~~~~~~~~~~~~~~~~~~
This expert-level tutorial demonstrates:

* Complex counterfactual predictions
* Recreating paper results on the Eraslan dataset
* Advanced model configurations
* Result analysis and validation

:download:`Download Notebook <Eraslan_CF_Tutorial.ipynb>`

.. raw:: html

    <div class="tutorial-nav">
        <a href="../installation.html">← Installation Guide</a>
        <a href="../api/index.html">API Reference →</a>
    </div>

    <style>
        .tutorials-header {
            background-color: #f8f9fa;
            padding: 2em;
            margin-bottom: 2em;
            border-radius: 5px;
            border-left: 5px solid #2980B9;
        }
        
        .tutorials-intro {
            font-size: 1.1em;
            line-height: 1.6;
            margin: 0;
            color: #444;
        }
        
        .tutorial-card {
            height: 100%;
            transition: transform 0.2s ease;
            margin-bottom: 1em;
        }
        
        .tutorial-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        
        h1, h2, h3, h4, h5, h6 {
            color: #2c3e50;
            margin-top: 1.5em;
            margin-bottom: 0.8em;
        }
        
        .note {
            background-color: #f8f9fa;
            padding: 1em;
            margin: 1em 0;
            border-left: 4px solid #2980B9;
        }
    </style>