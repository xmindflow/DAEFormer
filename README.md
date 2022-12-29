# DAEFormer

the DAEFormer is a U-Net like hierarchical pure transformer architecture.



## Updates
- 29 Dec., 2022: Initial release with arXiv.
- 27 Dec., 2022: Submitted to MIDL 2023 [Under Review].


## Citation
```
@article{azad2022daeformer,
  title={DAE-Former: Dual Attention-guided Efficient Transformer for Medical Image Segmentation},
  author={Azad, Reza and Arimond, René and Aghdam, Ehsan Khodapanah and Kazerouni, Amirhosein and Merhof, Dorit},
  journal={arXiv preprint arXiv:2212.13504},
  year={2022}
}
```

## How to use

The script train.py contains all the necessary steps for training the network. A list and dataloader for the Synapse dataset are also included.
To load a network, use the --module argument when running the train script (``--module <directory>.<module_name>.<class_name>``, e.g. ``--module networks.DAEFormer.DAEFormer``)





### Model weights
You can download the learned weights of the DAEFormer in the following table. 

Task | Dataset |Learned weights
------------ | -------------|----
Multi organ segmentation | [Synaps](http://www.isi.uu.nl/Research/Databases/DRIVE/) |[DAEFormer]()




### Query
All implementation done by Rene Arimond. For any query please contact us for more information.

```python
rene.arimond@lfb.rwth-aachen.de

```
