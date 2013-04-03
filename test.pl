# Before `make install' is performed this script should be runnable with
# `make test'. After `make install' it should work as `perl test.pl'

#########################
use Test;
BEGIN { plan tests => 1 };
#########################
print "Load libml-perl Modules.\n";
use ML::SVM;
use ML::SVM::Model;
ok(1);
#########################
print "Creating new SVM.\n";
my $svm = new ML::SVM();
ok(ref($svm) ne "", 1);
#########################
print "Creating new ML::SVM::Model.\n";
my $model = new ML::SVM::Model();
ok(ref($model) ne "", 1);
#########################
print "Read in a1a.model.\n";
$model->loadFile('a1a.model.bz2');
ok(1);
#########################
print "Predict on Test Set.\n";
$svm->predictFile('a1a.pedicted','a1a',$model);
ok(1);
#########################
print "Create New Model file.\n";
my $model_b = new ML::SVM::Model();
ok(ref($model_b) ne "", 1);
#########################
print "Create New SVM.\n";
my $svm_b = new ML::SVM();
ok(ref($svm_b) ne "", 1);
#########################
print "Load Test Set.\n";
$svm->read_problem_file($model_b,'a1a.t');
ok(1);
#########################
print "Load additional Vector.\n";
$svm->add_vector(-1,
  {'2'=>'1','8'=>'1','17'=>'1','19'=>'1','39'=>'1','40'=>'1','48'=>'1','63'=>'1','67'=>'1','73'=>'1','74'=>'1','76'=>'1','82'=>'1','87'=>'1'});
print 'This does not set gamma, so if not set run $model->set_gamma(1 / max_index);'."\n";
ok(1);
#########################
print "Train on Test Set.\n";
$svm->train($model_b);
ok(1);
#########################
print "Save Model File.\n";
$model->saveFile('a1a.model_b');
ok(1);
#########################
print "Predict on Single Vector.\n";
my $y_predict = $self->predict($model_b,
  {'4'=>'1','6'=>'1','16'=>'1','33'=>'1','35'=>'1','40'=>'1','48'=>'1','63'=>'1','67'=>'1','73'=>'1','74'=>'1','76'=>'1','80'=>'1','83'=>'1'});
ok(1);
#########################
# More methods exist for interacting with libml-svm
# see perlDoc for more information on these.
