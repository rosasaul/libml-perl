#!/usr/bin/perl
# Copyright (c) Saul Rosa 2012 GNU v3

my $VERSION = 0;

my $REVISIONS = '
0 Initial File
';

# Use Functions
use strict;
use ML::SVM;
use Getopt::Long;

my $help = "svm-train v$VERSION libsvm compatible perl native training CL tool

USAGE:
 svm-train [OPTIONS] training_set_file [model_file]

OPTIONS:
-help,h          Print this help menu
-svm,s           Set type of SVM (default C-SVC)
  0,c_svc      : Regularized support vector classification (standard algorithm)
  1,nu_svc     : Automatically regularized support vector classification
  2,one_class  : Select a hyper-sphere to maximize density
  3,epsilon_svr: Support vector regression robust to small (epsilon) errors
  4,nu_svr     : Support vector regression automatically minimize epsilon
-kernel,t        Set type of kernel function (default radial)
  0,linear     : u'*v
  1,polynomial : (gamma*u'*v + coef0)^degree
  2,rbf        : exp(-gamma*|u-v|^2)
  3,sigmoid    : tanh(gamma*u'*v + coef0)
-degree,d        Set degree in kernel function (default 3)
-gamma,g         Set gamma in kernel function (default 1/num_features)
-coef0,r         Set coef0 in kernel function (default 0)
-cost,c          Set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)
-nu,n            Set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)
-epsilon_svr,p   Set the epsilon in loss function of epsilon-SVR (default 0.1)
-epsilon_trm,e   Set tolerance of termination criterion (default 0.001)
-shrinking,h     Whether to use the shrinking heuristics, 0 or 1 (default 1)
-probability,b   Whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)
-weight,w        Set the parameter C of class i to weight*C, for C-SVC (default 1)
                 Usage : -w i:C -w i:C ...
-n_fold,v        n-fold cross validation mode
-quiet,q         No outputs
-threads         Disable/Enable threads, usage -nothreads/-threads (default Enabled)
-maxthr          Maximum concurrent thread count, default 2x Cores
-thrTune         Integer value to maximize thread performance, default 1440000
                 number of cols * number of rows each thread can handle,
                 will depend on IO bottleneck

libml-perl  Copyright (C) 2013  Saul Rosa
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
";

my $helpMenu;
my $kernel_type = 'rbf';
my $svm_type = 'c_svc';
my $degree = 3;
my $gamma;
my $coef0 = 0;
my $cost = 1;
my $nu = 0.5;
my $epsilon_svr = 0.1;
my $epsilon_trm = 0.001;
my $shrinking = 1;
my $probability = 0;
my @weights;
my $n_fold;
my $quiet;
my $threads = 1;
my $maxthr;
my $thrTune;
my $debug = 0;
my $debug_time = 0;

GetOptions(
  "help!"        => \$helpMenu,
  "kernel=s"     => \$kernel_type,
  "t=s"          => \$kernel_type,
  "svm=s"        => \$svm_type,
  "s=s"          => \$svm_type,
  "degree=s"     => \$degree,
  "gamma=s"      => \$gamma,
  "coef0=s"      => \$coef0,
  "r=s"          => \$coef0,
  "cost=s"       => \$cost,
  "c=s"          => \$cost,
  "nu=s"         => \$nu,
  "n=s"          => \$nu,
  "epsilon_svr=s"=> \$epsilon_svr,
  "p=s"          => \$epsilon_svr,
  "epsilon_trm=s"=> \$epsilon_trm,
  "e=s"          => \$epsilon_trm,
  "shrinking!"   => \$shrinking,
  "h!"           => \$shrinking,
  "probability=i"=> \$probability,
  "b=i"          => \$probability,
  "weight=s"     => \@weights,
  "n_fold=i"     => \$n_fold,
  "v=i"          => \$n_fold,
  "threads!"     => \$threads,
  "quiet!"       => \$quiet,
  "maxthr=i"     => \$maxthr,
  "thrTune=i"    => \$thrTune,
  "debug_time!"  => \$debug_time,
  "debug!"       => \$debug) or
die "Uknown option, use -help for additional information\n";

my $training_set_file = shift @ARGV;
my $model_file;
$model_file = shift @ARGV if @ARGV;

if($helpMenu or !$training_set_file){
  if(!$helpMenu and !$training_set_file){
    print STDERR "\n[31mNo training_set_file specified.[0m\n\n";
  }
  if($debug){print STDERR $REVISIONS;}
  print STDERR $help;
  exit;
}

## Main Functional ##
# Correct the name if decimal is picked
# SVM types
my %SVM_TYPES = (
  0             => 'c_svc',
  'c_svc'       => 'c_svc',
  1             => 'nu_svc',
  'nu_svc'      => 'nu_svc',
  2             => 'one_class',
  'one_class'   => 'one_class',
  3             => 'epsilon_svr',
  'epsilon_svr' => 'epsilon_svr',
  4             => 'nu_svr',
  'nu_svr'      => 'nu_svr'
);

# Kernel types
my %KERNEL_TYPES = (
  0            => 'linear',
  'linear'     => 'linear',
  1            => 'polynomial',
  'polynomial' => 'polynomial',
  2            => 'radial',
  'rbf'        => 'rbf',
  3            => 'sigmoid',
  'sigmoid'    => 'sigmoid'
)
;
if(!defined($KERNEL_TYPES{$kernel_type})){
  die "ERROR: Uknown kernel type : $kernel_type\n";
}
else{ $kernel_type = $KERNEL_TYPES{$kernel_type}; }
if(!defined($SVM_TYPES{$svm_type})){
  die "ERROR: Uknown svm type : $svm_type\n";
}
else{ $svm_type = $SVM_TYPES{$svm_type}; }

# Define Model
my $model = new ML::SVM::Model();

# Set Configs
$model->set_svm_type($svm_type) if defined $svm_type;
$model->set_kernel_type($kernel_type) if defined $kernel_type;
$model->set_degree($degree) if defined $degree;
$model->set_gamma($gamma) if defined $gamma;
$model->set_coef0($coef0) if defined $coef0;

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
$config{'threads'} = $threads if $threads;
$config{'maxthreads'} = $maxthr if $maxthr;
$config{'thrTune'} = $thrTune if $thrTune;
$config{'debug'} = $debug if $debug;
$config{'debug_time'} = $debug_time if $debug_time;

# Setup SVM
my $svm = new ML::SVM(\%config);

# Read in Problem
if($debug){ print STDERR "DBG: Read Problem\n"; }
$svm->read_problem_file($model,$training_set_file);

# Preform Training or Cross Validation
if($n_fold){ cross_validation($svm,$training_set_file); }
else{
  if($debug){ print STDERR "DBG: Train on Model File\n"; }
  $svm->train($model);
  #TODO Do it this way to cut Memory -> $svm->trainFile($training_set_file);
  if($debug){ print STDERR "DBG: Save Model File\n"; }
  if($model_file){ $model->saveFile($model_file); }
}

exit;

sub cross_validation {
  my $svm = shift;
  my $training_set_file = shift;
  my $total_correct = 0;
  my $total_error = 0;
  my $sumv = 0; my $sumy = 0;
  my $sumvv = 0; my $sumyy = 0; my $sumvy = 0;
  my @target;
  $svm->cross_validation($model,\@target);
  my $num_test_vectors = $svm->num_test_vectors();
  my @proby = $svm->problem_y_vectors();
  if($svm_type eq 'epsilon_svr' or $svm_type eq 'nu_svr'){
    for(my $i = 0; $i < $num_test_vectors; $i++){
      my $y = $proby[$i];
      my $v = $target[$i];
      $total_error += ($v - $y) * ( $v - $y);
      $sumv += $v;
      $sumy += $y;
      $sumvv += $v * $v;
      $sumyy += $y * $y;
      $sumvy += $v * $y;
    }
    print STDERR "Cross Validation Mean squared error = ".($total_error / $num_test_vectors)."\n";
    print STDERR "Cross Validation Squared correlation coefficient = ".(
      (($num_test_vectors * $sumvy - $sumv * $sumy) * ($num_test_vectors * $sumvy - $sumv * $sumy))/
      (($num_test_vectors * $sumvv - $sumv * $sumv) * ($num_test_vectors * $sumyy - $sumy * $sumy))
    )."\n";
  }
  else{
#    print STDERR "num_test_vectors $num_test_vectors\n";
    for(my $i = 0; $i < $num_test_vectors; $i++){
      if($target[$i] eq $proby[$i]){ $total_correct++; }
#      print STDERR "i $i target[i] $target[$i] proby[i] $proby[$i] total_correct $total_correct\n";
    }
    print STDERR "Cross Validation Accuracy = ".(100 * $total_correct / $num_test_vectors)."\n";
  }
}
