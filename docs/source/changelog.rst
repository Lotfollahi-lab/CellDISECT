==========
Changelog
==========

This file documents all notable changes to CellDISECT.

0.2.0b1 (2024-03-16)
------------------

* Beta release with compatibility updates
* Updates:
    * Compatible with Google Colab
    * Support for newer torch versions (>=2.1.0, <2.3.0)
    * Support for newer scvi-tools versions (>=1.0.0, <=1.3.0)
* Note:
    * This beta version is designed to work with modern environments while the main branch remains for reproducibility
    * Although this version introduces breaking changes, it is still backward compatible with the previous version (training and numerical process could be different due to different torch versions, hence slightly different results)

0.1.1 (2024-03-11)
------------------

* First public release
* Core functionality:
    * Cell state disentanglement
    * Covariate counterfactual predictions
    * Batch effect correction
* Documentation:
    * Basic API documentation
    * Installation guide
    * Tutorial notebooks
* Features:
    * Support for categorical covariates
    * Automatic cell type clustering
    * Integration with scanpy/anndata 