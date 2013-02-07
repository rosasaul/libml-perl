# Before `make install' is performed this script should be runnable with
# `make test'. After `make install' it should work as `perl test.pl'

#########################

use Test;
BEGIN { plan tests => 1 };

print "Load libml-perl Modules.\n";

use ML::SVM;
use ML::SVM::Model;

ok(1); # If we made it this far, we're ok.

#########################

print "Creating new ML::SVM.\n";
my $svm = new ML::SVM(Model => 'sample.model');
ok(ref($svm) ne "", 1);

print "Creating new ML::SVM::Model.\n";
my $model = new ML::SVM::Model();
ok(ref($model) ne "", 1);

print "Read in sample.model.\n";
$model->loadFile('sample.model');
ok(1);

print "Load sample test vectors.\n";
ok(1);

print "Check prediction of sample test vectors.\n";
ok(1);

print "Create new model object.\n";
ok(1);

print "Train on test vectors.\n";
ok(1);

#########################
# More methods exist for interacting with libml-svm
# see perlDoc for more information on these.
