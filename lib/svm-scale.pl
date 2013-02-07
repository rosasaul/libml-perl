#!/usr/bin/perl
# Copyright (c) Saul Rosa 2012 GNU v3

my $VERSION = 0;

my $REVISIONS = '
0 Initial File
';

# Use Functions
use strict;
use ML::SVM::Scale;
use Getopt::Long;

my $help = "svm-scale v$VERSION libsvm compatible perl native scaling CL tool

USAGE:
 svm-scale [OPTIONS] inputFile1 inputFile2 ... > output

OPTIONS:
-help    Print this help menu
-lower   lower x scaling limit, default -1
-upper   upper x scaling limit, default +1
-y       y scaling lower/upper limit, default no scaling
         Usage: -y y_lower y_upper
-save    save file of scaling parameters, default no saving
-restore restore file of scaling, used to retain scaling from previous
         scaling using -save, default no restore file
-output  Send output to a file, default is STDOUT
-maxthr  Specify maximum concurrent thread count, default 2x Cores
-thrTune Integer value to maximize thread performance, default 1440000
         number of cols * number of rows each thread can handle,
         will depend on IO bottleneck

libml-perl  Copyright (C) 2013  Saul Rosa
This program comes with ABSOLUTELY NO WARRANTY.
This is free software, and you are welcome to redistribute it
under certain conditions; see LICENSE.txt for details.
";

my $debug = 0;
my $helpMenu;
my $lower = -1;
my $upper = 1;
my @y;
my $save;
my $restore;
my $output;
my $maxthr;
my $thrTune;

GetOptions(
  "help!"     => \$helpMenu,
  "lower=s"   => \$lower,
  "upper=s"   => \$upper,
  "y=s{2}"    => \@y,
  "save=s"    => \$save,
  "restore=s" => \$restore,
  "output=s"  => \$output,
  "maxthr=i"  => \$maxthr,
  "thrTune=i" => \$thrTune,
  "debug!"    => \$debug) or die "Uknown option, use -help for additional information\n";
my @inputs = @ARGV;
if($helpMenu or !@inputs){
  if(!@inputs and !$helpMenu){
    print STDERR "\n[31mNo input files specified.[0m\n\n";
  }
  if($debug){print STDERR $REVISIONS;}
  print STDERR $help;
  exit;
}

# Main Functional
my %config;
$config{'lower'} = $lower if defined($lower);
$config{'upper'} = $upper if defined($upper);
$config{'ylower'} = $y[0] if @y;
$config{'yupper'} = $y[1] if @y;
$config{'maxthreads'} = $maxthr if defined($maxthr);
$config{'thrTune'} = $thrTune if defined($thrTune);
$config{'debug'} = $debug if $debug;
my $scale = new ML::SVM::Scale(\%config);

# Restore a scaling file if defined
$scale->restore($restore) if $restore;

# determine output
my $out = \*STDOUT;
if($output){ open $out, '>'.$output or die "Unable to open output file $output.\n"; }

# process input files
$scale->process($out,@inputs);

# Save off scaling config file
$scale->save($save) if $save;

exit;

