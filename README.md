# compi-graph-cut
School Project on Graph-Cut Image Segementation from the course Computational Imaging of IMT Atlantique

# Requirements:
- NetworkX
- PyMaxflow (if you want to use the library from the paper)

# Setup
## Install PyMaxflow

```
pip install PyMaxflow
```
Tested on linux, it works fine. For Windows, you need to get a C++ Compiler from Visual Studio (v14).

## Paper this project is based on

	"An Experimental Comparison of Min-Cut/Max-Flow Algorithms for Energy Minimization in Vision."
	Yuri Boykov and Vladimir Kolmogorov.
	In IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI), 
	September 2004
    http://dx.doi.org/10.1109/TPAMI.2004.60

    
    "Graph Cuts and Efficient N-D Image Segmentation."
    Boykov, Y., Funka-Lea, G.
    Int J Comput Vision 70, 109–131 
    (2006)
    https://doi.org/10.1007/s11263-006-7934-5

## Dataset used for experiments

    https://www.robots.ox.ac.uk/~vgg/data/iseg/