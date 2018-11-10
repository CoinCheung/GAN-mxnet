wget -c http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget -c http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget -c http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget -c http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz

gunzip train-images-idx3-ubyte.gz
gunzip train-labels-idx1-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
gunzip t10k-labels-idx1-ubyte.gz

mkdir -p images/train images/val
mkdir -p build && cd build
cmake ..
make
./main
