# Copyright (c) Saul Rosa 2012 GNU v3
# This program comes with ABSOLUTELY NO WARRANTY.
# This is free software, and you are welcome to redistribute it
# under certain conditions; see LICENSE.txt for details.


package Algorithm::ML::SVM::Scale;

our $VERSION = '0';
my $REVISIONS = '
0 Initial File
';

BEGIN{unshift @INC,"/home/srosa/perl" if caller;} #Adding local dir
use strict;
use threads;

sub new {
  my $class = shift;
  my $config_ref = shift;
  my %configs;
  my $self = \%configs;
  foreach my $item (keys %$config_ref){ $self->{$item} = $$config_ref{$item}; }
  bless($self,$class);
  if($self->{'debug'}){ print STDERR "DEBUG : Creating new Algorithm::ML::SVM::Scale\n"; }
  if(!defined($self->{'maxthreads'})){
    # Windows
    if(defined($ENV{'NUMBER_OF_PROCESSORS'})){
      $self->{'maxthreads'} = 2 * $ENV{'NUMBER_OF_PROCESSORS'};
    }
    # OSX
    elsif(!(-e '/proc/cpuinfo')){ $self->{'maxthreads'} = 2 * `sysctl -n hw.ncpu`; }
    # Linux/Unix
    else{ $self->{'maxthreads'} = 2 * `grep processor /proc/cpuinfo | wc -l`; }
    # Should do at least 2 threads if all else fails
    if($self->{'maxthreads'} < 2){ $self->{'maxthreads'} = 2; }
  }
  if(!defined($self->{'thrTune'})){ $self->{'thrTune'} = 1440000; }
  return $self;
}

sub restore {
  my $self = shift;
  my $file = shift;
  if($self->{'debug'}){ print STDERR "DEBUG : Restoring scaling file\n"; }
  open FH, '<'.$file or die "[31mUnable to open restore file $file. [0m\n";
  while(my $line = <FH>){
    if($line =~ /^x/){
      my $line = <FH>;
      chomp($line);
      my ($lower, $upper) = split(/\s+/,$line);
      $self->{'lower'} = $lower;
      $self->{'upper'} = $upper;
      while(<FH>){
        if($line =~ /^(\S+)\s+(\S+)\s+(\S+)/){
          my $x = 1 * $1;
          my $min = 1 * $2;
          my $max = 1 * $3;
          $self->{'xScaleSet'}[$x]{'min'} = $min;
          $self->{'xScaleSet'}[$x]{'max'} = $max;
        }
        elsif($line =~ /^y/){ last; }
      }
      if($line =~ /^y/){ redo; }
    }
    elsif($line =~ /^y/){
      my $line = <FH>;
      chomp($line);
      my ($ylower, $yupper) = split(/\s+/,$line);
      $self->{'ylower'} = $ylower;
      $self->{'yupper'} = $yupper;
      $line = <FH>;
      chomp($line);
      my ($ymin, $ymax) = split(/\s+/,$line);
      $self->{'ymin'} = $ymin;
      $self->{'ymax'} = $ymax;
    }
  }
  $self->{'restore'} = 1;
  close FH;
}

sub _process_restore {
  my $self = shift;
  my $yScale = shift;
  my @lines = @_;
  my @xScaleSetMax;
  my @xScaleSetMin;
  my $ymax;
  my $ymin;
  foreach my $line (@lines){
    my @items = split(/\s+/,$line);
    my $y = shift @items;
    if($yScale){
      if(!defined($ymax) or $y > $ymax){ $ymax = $y; }
      if(!defined($ymin) or $y < $ymin){ $ymin = $y; }
    }
    foreach my $item (@items){
      if($item =~ /(\d+):(\S+)/){
        my $x = 1 * $1; my $val = 1 * $2;
        if(!defined($xScaleSetMax[$x]) or $val > $xScaleSetMax[$x]){
          $xScaleSetMax[$x] = $val;
        }
        if(!defined($xScaleSetMin[$x]) or $val < $xScaleSetMin[$x]){
          $xScaleSetMin[$x] = $val;
        }
      }
      else{ warn "Found improperly formated data $item\n"; }
    }
  }
  return ($ymax,$ymin,(1 * @xScaleSetMin),@xScaleSetMin,@xScaleSetMax);
}

sub _reduce_restore {
  my $self = shift;
  my $yScale = shift;
  my $ymax = shift;
  my $ymin = shift;
  my $xScaleSetSize = shift;
  my @scaleSet = @_;
  for(my $x = 0; $x < $xScaleSetSize; $x++){
    my $min = $scaleSet[$x];
    my $max = $scaleSet[$x + $xScaleSetSize];
    if(!defined($self->{'xScaleSet'}[$x]{'max'}) or $max > $self->{'xScaleSet'}[$x]{'max'}){
      $self->{'xScaleSet'}[$x]{'max'} = $max;
    }
    if(!defined($self->{'xScaleSet'}[$x]{'min'}) or $min < $self->{'xScaleSet'}[$x]{'min'}){
      $self->{'xScaleSet'}[$x]{'min'} = $min;
    }
  }
}

sub _gen_restore {
  my $self = shift;
  my @inputs = @_;
  my $yScale = 0;
  my @threads;
  my @dump;
  if(defined($self->{'ylower'}) and defined($self->{'yupper'})){ $yScale++; }
  foreach my $input (@inputs){
    open FH, '<'.$input or die "^[[31mUnable to open input file $input. ^[[0m\n";
    while(my $line = <FH>){ chomp($line);
      if(!defined($self->{'dumpsize'})){
        my @items = split(/\s+/,$line);
        $self->{'dumpsize'} = int($self->{'thrTune'} / @items);
      }
      if(@dump < $self->{'dumpsize'}){ push @dump, $line; }
      else{
        if(@threads > $self->{'maxthreads'}){
          my $thread = shift @threads;
          $self->_reduce_restore($yScale,$thread->join());
        }
        push @threads, threads->create('_process_restore',$self,$yScale,@dump);
        @dump = ();
        push @dump, $line;
      }
    }
    close FH;
  }
  if(@dump){
    if(@threads > $self->{'maxthreads'}){
      my $thread = shift @threads;
      $self->_reduce_restore($yScale,$thread->join());
    }
    push @threads, threads->create('_process_restore',$self,$yScale,@dump);
  }
  foreach my $thread (@threads){
    $self->_reduce_restore($yScale,$thread->join());
  }
}

sub _process_dump {
  my $self = shift;
  my @lines = @_;
  my $out = '';
  foreach my $line (@lines){
    my @items = split(/\s+/,$line);
    my $y = $self->_scale_yval(shift @items);
    $out .= $y;
    foreach my $item (@items){
      if($item =~ /(\d+):(\S+)/){
        my $x = 1 * $1; my $val = 1 * $2;
        $out .= ' '.$x.':'.$self->_scale_xval($x,$val);
      }
      else{ warn "Found improperly formated data $item\n"; }
    }
    $out .= "\n";
  }
  return $out;
}

sub process {
  my $self = shift;
  my $out = shift;
  if($self->{'debug'}){ print STDERR "DEBUG : Processing logs\n"; }
  my @inputs = @_;
  my @threads;
  my @dump;
  if(!$self->{'restore'}){ $self->_gen_restore(@inputs); }
  foreach my $input (@inputs){
    open FH, '<'.$input or die "[31mUnable to open input file $input. [0m\n";
    while(my $line = <FH>){ chomp($line);
      if(!defined($self->{'dumpsize'})){
        my @items = split(/\s+/,$line);
        $self->{'dumpsize'} = int($self->{'thrTune'} / @items);
      }
      if(@dump < $self->{'dumpsize'}){ push @dump, $line; }
      else{
        if(@threads > $self->{'maxthreads'}){
          my $thread = shift @threads;
          print $out $thread->join();
        }
        push @threads, threads->create('_process_dump',$self,@dump);
        @dump = ();
        push @dump, $line;
      }
    }
    close FH;
  }
  if(@dump){
    if(@threads > $self->{'maxthreads'}){
      my $thread = shift @threads;
      print $out $thread->join();
    }
    push @threads, threads->create('_process_dump',$self,@dump);
  }
  foreach my $thread (@threads){
    print $out $thread->join();
  }
}

sub _scale_yval {
  my $self = shift;
  my $y = shift;
  if(!defined($self->{'ymin'}) or !defined($self->{'ymax'})){ return $y; }
  $y = ($y - $self->{'ymin'}) * ($self->{'yupper'} - $self->{'ylower'}) /
  ($self->{'ymax'} - $self->{'ymin'}) + $self->{'ymin'};
  return $y;
}

sub _scale_xval {
  my $self = shift;
  my $x = shift;
  my $val = shift;
  if($val eq $self->{'xScaleSet'}[$x]{'min'}){ return $self->{'lower'}; }
  elsif($val eq $self->{'xScaleSet'}[$x]{'max'}){ return $self->{'upper'}; }
  else{
    #print STDERR "$val $x $self->{'xScaleSet'}[$x]{'upper'} - $self->{'xScaleSet'}[$x]{'lower'}\n";
    $val = ($val - $self->{'xScaleSet'}[$x]{'min'}) *
    ($self->{'upper'} - $self->{'lower'}) /
    ($self->{'xScaleSet'}[$x]{'max'} - $self->{'xScaleSet'}[$x]{'min'}) +
    $self->{'lower'};
    return $val;
  }
}

sub save {
  my $self = shift;
  my $file = shift;
  if($self->{'debug'}){ print STDERR "DEBUG : Inside save off scale file\n"; }
  open FH, '>'.$file or die "[31mUnable to open save file $file. [0m\n";
  if(defined($self->{'ymin'}) and defined($self->{'ymax'})){
    print FH "y";
    print FH "\n".$self->{'ylower'}." ".$self->{'yupper'}."\n";
    print FH "\n".$self->{'ymin'}." ".$self->{'ymax'}."\n";
  }
  if(defined($self->{'xScaleSet'})){
    print FH "x";
    print FH "\n".$self->{'lower'}." ".$self->{'upper'}."\n";
    for(my $x = 0; $x < @{ $self->{'xScaleSet'} }; $x++){
      if(defined($self->{'xScaleSet'}[$x])){
        print FH "\n".$x." ".
        $self->{'xScaleSet'}[$x]{'min'}." ".
        $self->{'xScaleSet'}[$x]{'max'};
      }
    }
  }
  close FH;
}

1;

