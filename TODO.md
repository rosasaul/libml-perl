# Fixes
* SVM.pm need a variables overhaul to cut memory use

    Memory usage is not optimized, lots to be done still

* Thread svm-predict

    Module needs to be threaded, for bulk file processing

* svm-predict isn't optimized yet

* svr_probability is broken

    values returned are wrong

* precomputed kernel

    Module is missing

* Need to fix two pass threading

    Currently two pass results in 0 support vectors

# Updates
* Documentation

    Need to finish documenting everything in perldoc format

* Control-C or SIG{INT} catch to pause 

    Catch to allow pause of svm and dump memory to a file

