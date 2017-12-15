## Blind Dehazing Algorithms

This is an implementation of single image blind dehazing algorithms. We attempt to implement two papers, the ICCP '16 [paper](http://ieeexplore.ieee.org/document/7492870/) "*Blind dehazing using internal patch recurrence*", by Yuval Bahat & Michal Irani, and the CVPR '09 [paper](http://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=5567108) "*Single Image Haze Removal Using Dark Channel Prior*" by Kaiming He, Jian Sun, and Xiaoou Tang.

The implementation of "*Single Image Haze Removal Using Dark Channel Prior*" is complete. Run it using `./start_dark_channel.sh`. Use the `--help` flag for a detailed list of arguments. Hyperparameters can be adjusted in `dark_prior/config/constants.yml`.

The implementation of "*Blind dehazing using internal patch recurrence*" is in progress.

Some results have been added to `report.pdf`.
