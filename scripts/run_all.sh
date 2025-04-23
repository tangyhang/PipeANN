echo This script runs all the tests in the "tests" directory,
echo and output the results to the "data" directory.
mkdir data

bash $(dirname $0)/tests/fig11.sh
bash $(dirname $0)/tests/fig12.sh
bash $(dirname $0)/tests/fig13.sh
bash $(dirname $0)/tests/fig14.sh
bash $(dirname $0)/tests/fig15.sh
bash $(dirname $0)/tests/fig16.sh
bash $(dirname $0)/tests/fig17.sh
bash $(dirname $0)/tests/fig18.sh
