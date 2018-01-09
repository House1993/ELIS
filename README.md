# ELIS

ELIS is an interpretable shapelet-based classifier for time series classification.
It provides the efficient discovery for shapelet candidates which are the most discriminating time series subsequence for each class.
It also supports the fine-grained shapelet adjustment to achieve accurate time series classification.

Our novel approach yields significant improvements to all three of the important metrics the industry is now using for performance evaluation: (1) accuracy, (2)computational efficiency, and (3) interpretability.
Moreover, unlike previous methods, ELIS has fewer parameters to tune.

<!-- This page supports our publication: -->

<!-- Efficient Learning Interpretable Shapelets for Accurate Time Series Classification, ICDE 2018 IEEE International Conference on Data Engineering, Zicheng Fang, Peng Wang, Wei Wang -->

## Building

ELIS is built by [CMake](https://cmake.org/). To build ELIS, execute following commands in the src directory:

```
mkdir run
mkdir run/data
cmake .
make
```

## How to use

1.discover shapelet candidates from training time series:
```
run/discover ts_length training_file ts_number
```

2.adjust shapelets and train classifier:
```
run/adjust train ts_length training_file ts_number alpha lambda regularization learning_rate
```

3.predict for test time series
```
run/adjust test ts_length testing_file ts_number
```

## Example

We show an example of training and testing with Beef dataset in [UCR Benchmark](http://www.cs.ucr.edu/~eamonn/time_series_data/).
```
cd run
mkdir beef
cd beef
pwd
../discover 470 ../data/Beef/Beef_TRAIN 30 > dout
../adjust train 470 ../data/Beef/Beef_TRAIN 30 -25 0.01 7600 0.01 > tout
../adjust test 470 ../data/Beef/Beef_TEST 30 -25 > rout
cd ..
```

Our approach ourputs 3 txt files:
```
init.txt - the discovered shapelet candidates for each class
learned.txt - the shapelets for each class
result.txt - testing results
```

We provide a script to run tests on 15 datasets
```
./script
```
