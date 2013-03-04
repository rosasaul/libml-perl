# Copyright (c) Saul Rosa 2012 GNU v3
# This program comes with ABSOLUTELY NO WARRANTY.
# This is free software, and you are welcome to redistribute it
# under certain conditions; see LICENSE.txt for details.

package Algorithm::ML::SVM::Model;

our $VERSION = '0';
my $REVISIONS = '
0 Initial File
';

BEGIN{unshift @INC,"/home/srosa/perl" if caller;} #Adding local dir
use strict;

sub new {
  my $class = shift;
  my $configs_ref = shift;
  my %configs;
  if($configs_ref){ %configs = %$configs_ref; }
  my @SV;
  my @sv_coef;
  my @probA; my @probB;
  my $self = {
    'debug'        => $configs{'debug'},
    # coefficients for SVs in decision functions (sv_coef[k-1][l])
    'sv_coef'      => \@sv_coef,
    # constants in decision functions (rho[k*(k-1)/2])
    'rho'          => '',
    # pariwise probability information
    'probA'        => \@probA,
    # pariwise probability information
    'probB'        => \@probB,
    # label of each class (label[k])
    'label'        => '',
    # Support Vectors
    'SV'           => \@SV,
    'compress'     => $configs{'compress'}
  };
  bless($self,$class);
  my @rho; $self->{'param'}{'rho'} = \@rho;
  $self->{'param'}{'coef0'} = 0;
  return $self;
}

sub saveFile {
  my $self = shift;
  my $file = shift;
  my $fh;
  if($self->{'compress'}){
    use IO::Compress::Bzip2 qw(bzip2 $Bzip2Error);
    if(!($file =~ /\.bz2$/)){ $file .= '.bz2'; } # Add the bz2 end if not already
    $fh = new IO::Compress::Bzip2 $file or die "bunzip2 failed: $Bzip2Error\n";
  }
  else{ open $fh, '>'.$file or die "Unable to open file $file!\n"; }
  print $fh 'svm_type '.$self->{'param'}{'svm_type'}."\n";
  print $fh 'kernel_type '.$self->{'param'}{'kernel_type'}."\n";
  print $fh 'gamma '.$self->{'param'}{'gamma'}."\n" if defined $self->{'param'}{'gamma'};
  print $fh 'coef0 '.$self->{'param'}{'coef0'}."\n" if defined $self->{'param'}{'coef0'};
  print $fh 'nr_class '.$self->{'param'}{'nr_class'}."\n" if defined $self->{'param'}{'nr_class'};
  print $fh 'total_sv '.(1 * @{$self->{'SV'}})."\n" if defined $self->{'SV'};
  print $fh 'rho '.join(' ',@{$self->{'param'}{'rho'}})."\n" if defined $self->{'param'}{'rho'};
  print $fh 'label '.join(' ',@{$self->{'param'}{'label'}})."\n" if defined $self->{'param'}{'label'};
  print $fh 'probA '.join(' ',@{$self->{'probA'}})."\n" if @{ $self->{'probA'} };
  print $fh 'probB '.join(' ',@{$self->{'probB'}})."\n" if @{ $self->{'probB'} };
  print $fh 'nr_sv '.join(' ',@{$self->{'param'}{'nr_sv'}})."\n" if defined $self->{'param'}{'nr_sv'};
  print $fh "SV\n";
  for(my $row = 0; $row < @{$self->{'SV'}}; $row++){
    my @items;
    for(my $class = 0; $class < $self->{'param'}{'nr_class'} - 1; $class++){
      push @items, $self->{'sv_coef'}[$row][$class];
    }
    my @indexes = keys %{ $self->{'SV'}[$row] };
    @indexes = sort {$a <=> $b} @indexes;
    foreach my $index (@indexes){
      push @items, $index.':'.$self->{'SV'}[$row]{$index};
    }
    print $fh join(' ',@items)."\n";
  }
  close $fh;
}

sub loadFile {
  my $self = shift;
  my $file = shift;
  if(!(-e $file)){ die "File $file does not exist.\n"; }
  my $result = join(' ',`file "$file"`);
  my $fh;
  if($result =~ /bzip2/){
    use IO::Uncompress::Bunzip2 qw(bunzip2 $Bunzip2Error) ;
    $fh = new IO::Uncompress::Bunzip2 $file or die "bunzip2 failed: $Bunzip2Error\n";
  }
  elsif($result =~ /gzip/){
    use IO::Uncompress::Gunzip qw(gunzip $GunzipError) ;
    $fh = new IO::Uncompress::Gunzip $file or die "Gunzip failed: $GunzipError\n";
  }
  elsif($result =~ /ASCII/){
    open $fh, "<".$file or die "Unable to open model_file $file.\n";
  }
  else{ die "Unhandled log type $result from $file"; }
  while(<$fh>){ chomp;
    if(/svm_type (\S+)/){ $self->{'param'}{'svm_type'} = $1; }
    elsif(/kernel_type (\S+)/){ $self->{'param'}{'kernel_type'} = $1; }
    elsif(/degree (\S+)/){ $self->{'param'}{'degree'} = 1 * $1; }
    elsif(/gamma (\S+)/){ $self->{'param'}{'gamma'} = 1 * $1; }
    elsif(/coef0 (\S+)/){ $self->{'param'}{'coef0'} = 1 * $1; }
    elsif(/nr_class (\S+)/){ $self->{'param'}{'nr_class'} = 1 * $1; }
    elsif(/total_sv (\S+)/){ $self->{'param'}{'total_sv'} = 1 * $1; }
    elsif(/rho (.*)/){ my @rho = split(/\s+/,$1); $self->{'param'}{'rho'} = \@rho; }
    elsif(/label (.*)/){ my @label = split(/\s+/,$1); $self->{'param'}{'label'} = \@label; }
    elsif(/nr_sv (.*)/){ my @nr_sv = split(/\s+/,$1); $self->{'param'}{'nr_sv'} = \@nr_sv; }
    elsif(/probA (.*)/){ my @probA = split(/\s+/,$1); $self->{'probA'} = \@probA; }
    elsif(/probB (.*)/){ my @probB = split(/\s+/,$1); $self->{'probB'} = \@probB; }
    elsif(/SV/){
      my $row;
      while(<FH>){ chomp;
        my @sv = split(/\s+/);
        for(my $class = 0; $class < $self->{'param'}{'nr_class'} - 1; $class++){
          $self->{'sv_coef'}[$row][$class] = shift @sv;
        }
        foreach my $item (@sv){
          my ($index,$value) = split(/:/,$item);
          $self->{'SV'}[$row]{$index} = $value;
        }
        $row++;
      }
    }
  }
  close $fh;
}

sub sv {
  my $self = shift;
  my $row = shift;
  return $self->{'SV'}[$row];
}

sub param_copy {
  my $self = shift;
  my $model = shift;
  foreach my $key (keys %{ $self->{'param'} }){
    $model->{'param'}{$key} = $self->{'param'}{$key};
  }
}

sub count { my $self = shift; return @{ $self->{'SV'} }; }
sub sv_coef {
  my $self = shift;
  my $row = shift;
  my $class = shift;
  return $self->{'sv_coef'}[$row][$class];
}
sub svm_type { my $self = shift; return $self->{'param'}{'svm_type'}; }
sub rho { my $self = shift; my $class = shift; return $self->{'param'}{'rho'}->[$class]; }
sub nr_sv { my $self = shift; my $class = shift; return $self->{'param'}{'nr_sv'}->[$class]; }
sub label { my $self = shift; my $class = shift; return $self->{'param'}{'label'}->[$class]; }
sub kernel_type { my $self = shift; return $self->{'param'}{'kernel_type'}; }
sub gamma { my $self = shift; return $self->{'param'}{'gamma'}; }
sub coef0 { my $self = shift; return $self->{'param'}{'coef0'}; }
sub degree { my $self = shift; return $self->{'param'}{'degree'}; }
sub nr_class { my $self = shift; return $self->{'param'}{'nr_class'}; }
sub sv_coef_class_vector {
  my $self = shift; my $class = shift;
  my @vector;
  for(my $row=0;$row<@{ $self->{'sv_coef'} };$row++){
    $vector[$row] = $self->{'sv_coef'}[$row][$class];
  }
  return @vector;
}

sub set_sv_coef{
  my $self = shift;
  my $row = shift; my $class = shift;
  $self->{'sv_coef'}[$row][$class] = shift;
}
sub set_svm_type { my $self = shift; $self->{'param'}{'svm_type'} = shift; }
sub set_kernel_type { my $self = shift; $self->{'param'}{'kernel_type'} = shift; }
sub set_degree { my $self = shift; $self->{'param'}{'degree'} = shift; }
sub set_gamma { my $self = shift;  $self->{'param'}{'gamma'} = shift; }
sub set_coef0 { my $self = shift; $self->{'param'}{'coef0'} = shift; }
sub set_nr_class { my $self = shift; $self->{'param'}{'nr_class'} = shift; }
sub set_rho { my $self = shift; my $class = shift; $self->{'param'}{'rho'}->[$class] = shift; }
sub set_label { my $self = shift; my $class = shift; $self->{'param'}{'label'}->[$class] = shift; }
sub set_SV { my $self = shift; my $row = shift; $self->{'SV'}[$row] = shift; }
sub set_nSV { my $self = shift; my $row = shift; $self->{'param'}{'nr_sv'}[$row] = shift; }
sub set_probA { my $self = shift; my $row = shift; $self->{'probA'}[$row] = shift; }
sub set_probB { my $self = shift; my $row = shift; $self->{'probB'}[$row] = shift; }
sub set_nr_sv { my $self = shift; my $class = shift; $self->{'param'}{'nr_sv'}->[$class] = shift; }

1;
