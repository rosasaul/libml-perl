# libml-perl
This is a perl native SVM library based on 
[libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm). Code was a direct port of 
libsvm. Then it was updated for perl (e.g. removing memory allocation stuff 
and changing some of the OO stuff), and updated for threading.

## Dependencies

Perl compiled with threading, I suggest 5.14 or better, others have memory 
leaks and bugs in threads.

Beyond the obvious (threads)...

* Perl Math::Trig
 Used for calculating kernel functions.
* Perl Getopt::Long
 Used for the options entries
* Perl Storable
 Used to save off current memory into a file.
* Perl IO::Compress::Bzip2
 Used to compress out saved memory files.

## Installation
To install follow standard perl module install

    $ perl Makefile.PL
    $ make
    $ make test
    $ make install

## Usage with libsvm compatible scripts
Each script has a -help option to print out a help menu. All the options are libsvm compatible except for the weighted classes (See further down for more information).

Options are handeled using Getopt::Long so if their are no conflicts most options can be shortened. Meaning since no other option begins with the letter 'h' you can use -h instead of -help. Same applies to all other options.

Weighted Classes are handeled in libsvm with the options '-wi C' where 'w' is the option prefix, 'i' is replaced with the feature number, and 'C' is the weight. For svm-train.pl to keep things consistent with Getopt::Long formatting the option is formated '-w i:C -w i:C ...'.

Use svm-scale.pl to scale files/features to the desired ranges. Use svm-predict.pl to predict values of a test file using a defined model file. Finally use svm-train.pl to train on data set files, as well as cross validation and auto finding coefficients.

Rename removing the '.pl' from each script or use a symbolic link to make them apear the same as the libsvm binaries.

Usage:

    $ ./svm-scale [OPTIONS] inputFile1 inputFile2 ... > output
    $ ./svm-predict [OPTIONS] test_file model_file output_file
    $ ./svm-train.pl [OPTIONS] training_set_file [model_file]

## Usage as a Perl module
ML::SVM is the main SVM engine and will contain the associated training data. ML::SVM::Model is the model file which specifies the type of svm and kernel to use as well as coefficients. When Training is complete ML::SVM::Model will also contain the Support Vectors.

Create a new SVM object.
The hash pointer is not required but default settings will be selected when no conditions are set.
```perl
my %config;
$config{'cost'} = $cost if defined $cost;
$config{'nu'} = $nu if defined $nu;
$config{'epsilon_svr'} = $epsilon_svr if defined $epsilon_svr;
$config{'epsilon_trm'} = $epsilon_trm if defined $epsilon_trm;
$config{'shrinking'} = $shrinking if defined $shrinking;
$config{'probability'} = $probability if defined $probability;
$config{'nr_weights'} = 1 * @weights if @weights;
$config{'n_fold'} = $n_fold if defined $n_fold;
my %hWeights; $config{'weights'} = \%hWeights;
foreach my $weight (@weights){
  my ($label,$value) = split(/:/,$weight);
  $config{'weights'}{$label} = $value;
}
$config{'quiet'} = $quiet if $quiet;
$config{'maxthreads'} = $maxthr if $maxthr;
$config{'thrTune'} = $thrTune if $thrTune;

my $svm = new ML::SVM(\%config);
```

Create a new SVM Model. These contain the parameters, the Support Vectors and the svm type and kernel type information.
```perl
my $model = new ML::SVM::Model();

# Set Configs
$model->set_svm_type($svm_type) if defined $svm_type;
$model->set_kernel_type($kernel_type) if defined $kernel_type;
$model->set_degree($degree) if defined $degree;
$model->set_gamma($gamma) if defined $gamma;
$model->set_coef0($coef0) if defined $coef0;
```

Add new data to the SVM.
```perl
my %xSet;
# Assign each feature and it's value
$xSet{$feature} = $value;

my $row = $svm->add_vector(\%xSet);
```

Train the SVM using after loading data and establishing an SVM Model
```perl
$svm->train($model);
```

Predict the values of a data set using an SVM Model
```perl
# @dec_values is the raw prediction numbers, and is optional
my $y_predicted = $self->predict($model,\%xSet,\@dec_values);
```

Save off a model file
```perl
$model->saveFile('model_file_name');
```

## Bugs
For bug reports and issues use the [github issues page](https://github.com/rosasaul/libml-perl/issues), try to make sure I'll be able to reproduce the issue.
* Not a bug per-se, but perl has full precision numbers so when using the svm epsilon_svr models may not match libsvm exactly.

## Copyright and Licence
libml-perl is a libsvm compatible threaded SVM library writen in native perl.
Copyright (C) 2013 Saul Rosa

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

## Author Information
This module was written by Saul Rosa <rosa.saul@megaframe.org>.

I'm an electrical engineer who uses a lot of code to get my work done. The 
modules and wrappers were written to address the shortcommings of libsvm 
(it's non-threaded and I need to be able to pause the job and resume later on 
another machine) while making it perl native so I can integrate it into other 
data analysis tools.

## Acknowledgements
Thanks to the authors of [libsvm](http://www.csie.ntu.edu.tw/~cjlin/libsvm).

Thanks to the author of the library [Algorith::SVM](http://search.cpan.org/~lairdm/Algorithm-SVM-0.13/lib/Algorithm/SVM.pm),
which was used as a basis for the design of this module.

