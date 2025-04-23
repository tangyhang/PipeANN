export ADDITIONAL_DEFINITIONS="-DDYN_PIPE_WIDTH"

mkdir build
cd build
cmake ..
make -j
