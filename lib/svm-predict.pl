#!/usr/bin/perl
# Copyright (c) Saul Rosa 2012 GNU v3

my $VERSION = 0;

my $REVISIONS = '
0 Initial File
';

BEGIN{unshift @INC,"/home/srosa/perl";} #Adding local dir

# Use Functions
use strict;
use Algorithm::ML::SVM;
use Getopt::Long;

my $help = "svm-predict v$VERSION libsvm compatible perl native prediction CL tool

USAGE:
 svm-predict [OPTIONS] test_file model_file output_file

OPTIONS:
-help           Print this help menu
-probability,b  Probability Estimates, whether to predict probability estimates,
                0 or 1, default 0
                for one-class SVM only 0 is supported
-maxthr         Specify maximum concurrent thread count, default 2x Cores
-thrTune        Integer value to maximize thread performance, default 1440000
                number of cols * number of rows each thread can handle,
                will depend on IO bottleneck

libml-perl  Copyright (C) 2013  Saul Rosa
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
";

my $probability = 0;
my $threads = 1;
my $maxthr;
my $thrTune;
my $debug = 0;
my $helpMenu;

GetOptions(
  "help!"        => \$helpMenu,
  "b!"           => \$probability,
  "probability!" => \$probability,
  "threads!"     => \$threads,
  "maxthr=i"     => \$maxthr,
  "thrTune=i"    => \$thrTune,
  "debug!"       => \$debug) or
die "Uknown option, use -help for additional information\n";

my $test_file = shift @ARGV;
my $model_file = shift @ARGV;
my $output_file = shift @ARGV;

if($helpMenu){
  if($debug){print STDERR $REVISIONS;}
  print STDERR $help;
  exit;
}

if(!$test_file or !$model_file or !$output_file){
  print STDERR "\n[31m";
  if(!$test_file){ print STDERR "No test_file specified.\n"; }
  if(!$model_file){ print STDERR "No model_file specified.\n"; }
  if(!$output_file){ print STDERR "No output_file specified.\n"; }
  print STDERR "[0m\n";
  print STDERR $help;
  exit;
}

open my $out, '>'.$output_file or die "Unable to open output_file $output_file.\n"; 

# Main Functional
my %config;
$config{'threads'} = $threads;
$config{'probability'} = $probability;
$config{'debug'} = $debug if $debug;

# Init the model
my $model = new Algorithm::ML::SVM::Model({'debug' => $debug});
$model->loadFile($model_file);

# Predict the output
my $svm = new Algorithm::ML::SVM(\%config);
$svm->predictFile($out,$test_file,$model);

exit;

