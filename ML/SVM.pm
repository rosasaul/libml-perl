# Copyright (c) Saul Rosa 2012 GNU v
# This program comes with ABSOLUTELY NO WARRANTY.
# This is free software, and you are welcome to redistribute it
# under certain conditions; see LICENSE.txt for details.

package Algorithm::ML::SVM;

use strict;
#use warnings;

BEGIN{unshift @INC,"/home/srosa/perl" if caller;} #Adding local dir

use Math::Trig;
use Algorithm::ML::SVM::Model;
use threads;

our $VERSION = '1';
my $REVISIONS = '
0 Initial File
1 Updated to threading in train()
';

sub new {
  my ($class,$config_ref) = @_;
  my %config;
  if($config_ref){
    foreach my $item (keys %$config_ref){ $config{$item} = $$config_ref{$item}; }
  }
  my $self = \%config;
  bless($self,$class);
  if($config{'threads'} and !defined($self->{'maxthreads'})){
    # Windows
    if(defined($ENV{'NUMBER_OF_PROCESSORS'})){
      $self->{'maxthreads'} = 2 * $ENV{'NUMBER_OF_PROCESSORS'};
    }
    # OSX
    elsif(!(-e '/proc/cpuinfo')){ $self->{'maxthreads'} = 2 * `sysctl -n hw.ncpu`; }
    # Linux/Unix
    else{ $self->{'maxthreads'} = 4 * `grep processor /proc/cpuinfo | wc -l`; }
    # Should do at least 2 threads if all else fails
    if($self->{'maxthreads'} < 2){ $self->{'maxthreads'} = 2; }
  }
  if($config{'threads'} and !defined($self->{'thrTune'})){ $self->{'thrTune'} = 4500; }
  $self->initCoef;
  return $self;
}

sub set_cost { my ($self,$cost) = @_; $self->{'cost'} = $cost; }
sub set_nu { my ($self,$nu) = @_; $self->{'nu'} = $nu; }
sub set_epsilon_svr { my ($self,$epsilon_svr) = @_; $self->{'epsilon_svr'} = $epsilon_svr; }
sub set_epsilon_trm { my ($self,$epsilon_trm) = @_; $self->{'epsilon_trm'} = $epsilon_trm; }

sub train {
  my ($self,$model,$save_file) = @_;
  my @features = keys %{ $self->{'keySetHash'} };
  $self->set_types($model);
  $self->{'keySet'} = \@features;
#  $self->loger("Starting Trainer");
  if($model->svm_type eq 'one_class' or
    $model->svm_type eq 'epsilon_svr' or
    $model->svm_type eq 'nu_svr'){
    $model->set_nr_class(2);
    if($self->{'probability'} and 
      ($model->svm_type eq 'epsilon_svr' or
       $model->svm_type eq 'nu_svr')){
      $model->{'probA'}->[0] = $self->svr_probability($model);
    }
    $self->{'Cn'} = 0;
    $self->{'Cp'} = 0;
#    $self->loger("Train One");
    my @alpha = $self->train_one($model);
    my $vector = 0;
    my $fh;
    if($save_file){
      use IO::Compress::Bzip2 qw(bzip2 $Bzip2Error);
      if(!($save_file =~ /\.bz2$/)){ $save_file .= '.bz2'; } # Add the bz2 end if not already
      $fh = new IO::Compress::Bzip2 $save_file or die "bunzip2 failed: $Bzip2Error\n";
    }
    for(my $i = 0; $i < $self->{'problem'}{'count'}; $i++){
      if(abs($alpha[$i]) > 0){
        $model->{'SV'}[$vector] = $self->{'problem'}{'x'}[$i];
        $model->set_sv_coef($vector,0,$alpha[$i]);
        $vector++;
        if($save_file){
          my @items;
          push @items, $self->{'problem'}{'y'}[$i];
          my @indexes = keys %{ $self->{'problem'}{'x'}[$i] };
          @indexes = sort {$a <=> $b} @indexes;
          foreach my $index (@indexes){
            push @items, $index.':'.$self->{'problem'}{'x'}[$i]{$index};
          }
          print $fh join(' ',@items)."\n";
        }
      }
    }
    if($save_file){ close $fh; }
  }
  else{
    # classification
    my $l = $self->{'problem'}{'count'};
    my $nr_class;
    my @label; my @start; my @count; my @perm;
    # group training data of the same class
    $self->group_classes(\$nr_class,\@label,\@start,\@count,\@perm);
    if($nr_class == 1 and !$self->{'quiet'}){
      print STDERR "WARNING: training data in only one class. See README for details.\n";
    }
    my @x;
    for(my $i=0;$i < $l; $i++){
      $x[$i] = $self->{'problem'}{'x'}[$perm[$i]];
    }
    # calculate weighted C
    my @weighted_C;
    for(my $i=0; $i < $nr_class; $i++){
      $weighted_C[$i] = $self->{'cost'};
    }
    foreach my $cur_label (keys %{ $self->{'weights'} }){
      my $j;
      for($j=0;$j < $nr_class;$j++){
        if($cur_label eq $label[$j]){ last; }
      }
      if($j == $nr_class and !$self->{'quiet'}){
        print STDERR "WARNING: class label ".$cur_label." specified in weight is not found\n";
      }
      else{ $weighted_C[$j] = $self->{'weights'}{$cur_label}; }
    }
    # train k*(k-1)/2 models
    my @nonzero;
    for(my $i=0;$i < $l; $i++){ $nonzero[$i] = 0; } #Set to False

    my @alpha;
    my @probA; my @probB;
    my @rho;
    my $p = 0;
    for(my $i=0;$i<$nr_class;$i++){
      for(my $j=$i + 1;$j < $nr_class;$j++){
        my $sub_prob = Algorithm::ML::SVM->new();
        my $si = $start[$i]; my $sj = $start[$j];
        my $ci = $count[$i]; my $cj = $count[$j];
        $sub_prob->{'problem'}{'count'} = $ci + $cj;
        for(my $k = 0; $k < $ci; $k++){
          $sub_prob->{'problem'}{'x'}[$k] = $x[$si + $k];
          $sub_prob->{'problem'}{'y'}[$k] = 1;
        }
        for(my $k = 0; $k < $cj; $k++){
          $sub_prob->{'problem'}{'x'}[$ci + $k] = $x[$sj + $k];
          $sub_prob->{'problem'}{'y'}[$ci + $k] = -1;
        }
        $self->param_copy($sub_prob);
        $sub_prob->{'Cp'} = $weighted_C[$i];
        $sub_prob->{'Cn'} = $weighted_C[$j];
        if($self->{'probability'}){
          my $A; my $B;
          $sub_prob->binary_svc_probability(\$A,\$B,$model);
          $probA[$p] = $A; $probB[$p] = $B;
        }
        my @alphap = $sub_prob->train_one($model);
        $alpha[$p] = \@alphap;
        for(my $k = 0;$k < $ci; $k++){
          if(!$nonzero[$si + $k] and abs($alpha[$p][$k]) > 0){ $nonzero[$si + $k] = 1; } #Set to True
        }
        for(my $k = 0;$k < $cj; $k++){
          if(!$nonzero[$sj + $k] and abs($alpha[$p][$ci + $k]) > 0){ $nonzero[$sj + $k] = 1; }
        }
        $rho[$p] = $sub_prob->{'rho'};
        $p++;
      }
    }
    # build output
    $model->set_nr_class($nr_class);
    for(my $i = 0;$i < $model->nr_class; $i++){ $model->set_label($i,$label[$i]); }
    for(my $i = 0;$i < $model->nr_class * ($model->nr_class - 1) / 2; $i++){
      $model->set_rho($i,$rho[$i]); #Todo Get back rho from each sub problem
    }
    if($self->{'probability'}){
      for(my $i = 0;$i < $nr_class * ($nr_class - 1) / 2;$i++){
        $model->set_probA($i,$probA[$i]);
        $model->set_probB($i,$probB[$i]);
      }
    }
    my $total_sv = 0;
    my @nz_count;
    for(my $i = 0;$i < $nr_class;$i++){
      my $nSV = 0;
      for(my $j = 0;$j < $count[$i];$j++){
        if($nonzero[$start[$i]+$j]){
          $nSV++;
          $total_sv++;
        }
      }
      $model->set_nSV($i,$nSV);
      $nz_count[$i] = $nSV;
    }
    if(!$self->{'quiet'}){ print STDERR "Total nSV = $total_sv\n"; }

    $p = 0;
    my $fh;
    if($save_file){
      use IO::Compress::Bzip2 qw(bzip2 $Bzip2Error);
      if(!($save_file =~ /\.bz2$/)){ $save_file .= '.bz2'; } # Add the bz2 end if not already
      $fh = new IO::Compress::Bzip2 $save_file or die "bunzip2 failed: $Bzip2Error\n";
#      print $fh "ORIGINAL_SV\n";
    }
    for(my $i = 0;$i < $l;$i++){
      if($nonzero[$i]){
        $model->set_SV($p++,$x[$i]);
        if($save_file){
          my @items;
          push @items, $self->{'problem'}{'y'}[$i];
          my @indexes = keys %{ $self->{'problem'}{'x'}[$i] };
          @indexes = sort {$a <=> $b} @indexes;
          foreach my $index (@indexes){
            push @items, $index.':'.$self->{'problem'}{'x'}[$i]{$index};
          }
          print $fh join(' ',@items)."\n";
        }
      }
    }
    if($save_file){ close $fh; }
    my @nz_start; $nz_start[0] = 0;
    for(my $i = 1;$i < $nr_class;$i++){
      $nz_start[$i] = $nz_start[$i - 1] + $nz_count[$i - 1];
    }
    $p = 0;
    for(my $i=0;$i < $nr_class;$i++){
      for(my $j = $i + 1;$j < $nr_class;$j++){
        # classifier (i,j): coefficients with
        # i are in sv_coef[j-1][nz_start[i]...],
        # j are in sv_coef[i][nz_start[j]...]
        my $si = $start[$i];
        my $sj = $start[$j];
        my $ci = $count[$i];
        my $cj = $count[$j];
        my $q = $nz_start[$i];
        for(my $k = 0;$k < $ci;$k++){
          if($nonzero[$si + $k]){ $model->set_sv_coef($q++,$j - 1,$alpha[$p][$k]); }
        }
        $q = $nz_start[$j];
        for(my $k = 0;$k < $cj;$k++){
          if($nonzero[$sj + $k]){ $model->set_sv_coef($q++,$i,$alpha[$p][$ci + $k]); }
        }
        $p++;
      }
    }
  }
  return $model;
}

sub binary_svc_probability{ # Cross-validation decision values for probability estimates
  my ($self,$probA,$probB,$model) = @_;
  my $n_fold = 5;
  my @perm; my @dec_values;

  # random shuffle
  my $l = $self->{'problem'}{'count'};
  for(my $i=0;$i<$l;$i++){ $perm[$i]=$i; }
  for(my $i=0;$i<$l;$i++){
    my $j = $i + int(rand($l - $i));
    ($perm[$i],$perm[$j]) = ($perm[$j],$perm[$i]);
  }
  for(my $i=0;$i<$n_fold;$i++){
    my $begin = $i * $l / $n_fold;
    my $end = ($i + 1) * $l / $n_fold;
    my $k = 0;
    my $sub_prob = Algorithm::ML::SVM->new();
    $sub_prob->{'problem'}{'count'} = $l - ($end - $begin);
    $self->param_copy($sub_prob);
    $k = 0;
    for(my $j=0;$j<$begin;$j++){
      $sub_prob->{'problem'}{'x'}[$k] = $self->{'problem'}{'x'}[$perm[$j]];
      $sub_prob->{'problem'}{'y'}[$k] = $self->{'problem'}{'y'}[$perm[$j]];
      $k++;
    }
    for(my $j=$end;$j<$l;$j++){
      $sub_prob->{'problem'}{'x'}[$k] = $self->{'problem'}{'x'}[$perm[$j]];
      $sub_prob->{'problem'}{'y'}[$k] = $self->{'problem'}{'y'}[$perm[$j]];
      $k++;
    }
    my $p_count=0; my $n_count=0;
    for(my $j=0;$j<$k;$j++){
      if($sub_prob->{'problem'}{'y'}[$j]>0){ $p_count++; }
      else{ $n_count++; }
    }
    if($p_count == 0 and $n_count == 0){
      for(my $j=$begin;$j<$end;$j++){ $dec_values[$perm[$j]] = 0; }
    }
    elsif($p_count > 0 and $n_count == 0){
      for(my $j=$begin;$j<$end;$j++){ $dec_values[$perm[$j]] = 1; }
    }
    elsif($p_count == 0 and $n_count > 0){
      for(my $j=$begin;$j<$end;$j++){ $dec_values[$perm[$j]] = -1; }
    }
    else{
      $sub_prob->{'probability'} = 0;
      $sub_prob->{'cost'} = 1;
      $sub_prob->{'nr_weight'} = 2;
      $sub_prob->{'weights'}{1} = $sub_prob->{'Cp'};
      $sub_prob->{'weights'}{-1} = $sub_prob->{'Cn'};
      my $sub_model = Algorithm::ML::SVM::Model->new();
      $model->param_copy($sub_model);
      $sub_prob->train($sub_model);
      for(my $j=$begin;$j<$end;$j++){
        $sub_prob->predict($sub_model,$self->{'problem'}{'x'}[$perm[$j]],\@dec_values);
        #dec_values[perm[j]]
        # ensure +1 -1 order; reason not using CV subroutine
        $dec_values[$perm[$j]] *= $sub_model->label(0);
      }
    }
  }
  $self->sigmoid_train(\@dec_values,$probA,$probB,$model);
}

sub sigmoid_train{ # Platt's binary SVM Probablistic Output: an improvement from Lin et al.
  my ($self,$dec_values,$A,$B,$model) = @_;
  my $prior1 = 0; my $prior0 = 0;
  my $l = $self->{'problem'}{'count'};
  for(my $i=0;$i<$l;$i++){
    if($model->label($i) > 0){ $prior1 += 1; }
    else{ $prior0 += 1; }
  }
  my $max_iter = 100;   # Maximal number of iterations
  my $min_step = 1e-10; # Minimal step taken in line search
  my $sigma = 1e-12;    # For numerically strict PD of Hessian
  my $eps = 1e-5;
  my $hiTarget = ($prior1 + 1) / ($prior1 + 2);
  my $loTarget = 1 / ($prior0 + 2);
  my @t; 
  my $fApB; 
  my $p; my $q;
  my $h11; my $h22; my $h21;
  my $g1; my $g2;
  my $det;
  my $dA; my $dB; my $gd;
  my $stepsize;
  my $newA; my $newB; my $newf;
  my $d1; my $d2;
  my $iter;
  # Initial Point and Initial Fun Value
  $$A = 0; $$B = log( ($prior0 + 1) / ($prior1 + 1));
  my $fval = 0;
  for(my $i=0;$i<$l;$i++){
    if($model->label($i) > 0){ $t[$i]=$hiTarget; }
    else{ $t[$i]=$loTarget; }
    $fApB = $$dec_values[$i] * $$A + $$B;
    if($fApB >= 0){ $fval += $t[$i] * $fApB + log(1 + exp(-$fApB)); }
    else{ $fval += ($t[$i] - 1) * $fApB + log(1 + exp($fApB)); }
  }
  for($iter=0;$iter < $max_iter;$iter++){
    # Update Gradient and Hessian (use H' = H + sigma I)
    $h11 = $sigma; # numerically ensures strict PD
    $h22 = $sigma;
    $h21 = 0; $g1 = 0; $g2 = 0;
    for(my $i=0;$i<$l;$i++){
      $fApB = $$dec_values[$i] * $$A + $$B;
      if($fApB >= 0){
        $p = exp(-$fApB) / (1 + exp(-$fApB));
        $q = 1 / (1 + exp(-$fApB));
      }
      else{
        $p = 1 / (1 + exp($fApB));
        $q = exp($fApB) / (1 + exp($fApB));
      }
      $d2 = $p * $q;
      $h11 += $$dec_values[$i] * $$dec_values[$i] * $d2;
      $h22 += $d2;
      $h21 += $$dec_values[$i] * $d2;
      $d1 = $t[$i] - $p;
      $g1 += $$dec_values[$i] * $d1;
      $g2 += $d1;
    }
    # Stopping Criteria
    if(abs($g1) < $eps and abs($g2) < $eps){ last; }
    # Finding Newton direction: -inv(H') * g
    $det = $h11 * $h22 - $h21 * $h21;
    $dA = -($h22 * $g1 - $h21 * $g2) / $det;
    $dB = -(-$h21 * $g1 + $h11 * $g2) / $det;
    $gd = $g1 * $dA + $g2 * $dB;
    $stepsize = 1; # Line Search
    while($stepsize >= $min_step){
      $newA = $$A + $stepsize * $dA;
      $newB = $$B + $stepsize * $dB;
      # New function value
      $newf = 0;
      for(my $i=0;$i<$l;$i++){
        $fApB = $$dec_values[$i] * $newA + $newB;
        if($fApB >= 0){ $newf += $t[$i] * $fApB + log(1 + exp(-$fApB)); }
        else{ $newf += ($t[$i] - 1) * $fApB +log(1 + exp($fApB)); }
      }
      # Check sufficient decrease
      if($newf < $fval + 0.0001 * $stepsize * $gd){
        $$A = $newA; $$B = $newB; $fval = $newf;
        last;
      }
      else{ $stepsize = $stepsize / 2; }
    }
    if($stepsize < $min_step){
      if(!$self->{'quiet'}){ print STDERR "Line search fails in two-class probability estimates\n"; }
      last;
    }
  }
  if($iter >= $max_iter and !$self->{'quiet'}){
    print STDERR "Reaching maximal iterations in two-class probability estimates\n";
  }
}

sub param_copy{
  my ($self,$sub_prob) = @_;
  foreach my $key (keys %{ $self }){
    if($key eq 'problem'){ next; }
    $sub_prob->{$key} = $self->{$key};
  }
}

sub train_one {
  my ($self,$model) = @_;
  if(!$self->{'threads'}){ return $self->train_one_lin($model); }
  # Sort Vectors by class
  my %classes;
  my %sorted_vectors;
  my %cnt_per_group; # Count per grouping
  # sort vectors by class and count number of items per class
  for(my $i = 0; $i < $self->{'problem'}{'count'}; $i++){
    my $class = $self->{'problem'}{'y'}[$i];
    $classes{ $class }++;
    $sorted_vectors{ $class }[ $cnt_per_group{$class}++ ] = $i;
  }
  # Weighted selection of vectors using rand()
  my $maxgroups = $self->{'maxthreads'};
  if($self->{'n_fold'}){ $maxgroups = int($self->{'maxthreads'} / $self->{'n_fold'}); }
  my @groupings;
  my $group = 0;
  my $count = $self->{'problem'}{'count'};
  my @class_list = keys %classes; # make sure order stays the same when searching
  for(my $group = 0;$group < $maxgroups ;$group++){ $cnt_per_group{$group} = 0; }
  while($count){
    if($group >= $maxgroups){ $group = 0; }
    my $selection = rand(); # float 0.0 to 1.0
    foreach my $class (@class_list){
      if($selection < $classes{$class} / $self->{'problem'}{'count'}){
        if(!@{ $sorted_vectors{$class} }){ next; }
        my $k = int(rand(1 * @{ $sorted_vectors{$class} }));
        my $i = $sorted_vectors{$class}[$k];
        $groupings[$group]{'original_index'}[$cnt_per_group{$group}] = $i;
        $cnt_per_group{$group}++;
        splice(@{ $sorted_vectors{$class} },$k,1);
        last;
      }
      $selection -= $classes{$class} / $self->{'problem'}{'count'};
    }
    $group++; $count--;
  }
  # Spawn threads for each grouping
  my @threads;
  foreach my $group (@groupings){
    push @threads, threads->create('_handle_grouping',$self,$model,$group);
  }
  # Gather all Support Vectors from each sub classification
  my @keep_SVs;
  foreach my $thread (@threads){
    push @keep_SVs, $thread->join();
  }
  # Remove uneeded vectors from main problem set
  my @x; my @y; my $k = 0;
  foreach my $index (@keep_SVs){
    $x[$k] = $self->{'problem'}{'x'}[$index];
    $y[$k] = $self->{'problem'}{'y'}[$index];
    $k++;
  }
  $self->{'problem'}{'count'} = 1 * @x;
  $self->{'problem'}{'x'} = \@x;
  $self->{'problem'}{'y'} = \@y;
  # Run Solver on subsection of vectors
  return $self->train_one_lin($model);
}

sub _handle_grouping {
  my ($self,$model,$group) = @_;
  my $sub_prob = Algorithm::ML::SVM->new();
  $self->param_copy($sub_prob);
  $sub_prob->{'threads'} = 0;
  $sub_prob->{'problem'}{'count'} = 1 * @{ $$group{'original_index'} };
  my $k = 0;
  for(my $j=0;$j < $sub_prob->{'problem'}{'count'};$j++){
    my $i = $$group{'original_index'}[$j];
    $sub_prob->{'problem'}{'x'}[$j] = $self->{'problem'}{'x'}[$i];
    $sub_prob->{'problem'}{'y'}[$j] = $self->{'problem'}{'y'}[$i];
  }
  my $sub_model = Algorithm::ML::SVM::Model->new();
  $model->param_copy($sub_model);
  my @alpha = $sub_prob->train_one_lin($sub_model);
  # output SVs
  my @SVs;
  for(my $row=0; $row < $sub_prob->{'problem'}{'count'}; $row++){
    if(abs($alpha[$row]) > 0){
      push @SVs, $$group{'original_index'}[$row];
    }
  }
  return @SVs;
}

sub train_one_lin {
  my ($self,$model) = @_;
  my @alpha;
  if($model->svm_type eq 'c_svc'){ @alpha = $self->solve_c_svc($model); }
  elsif($model->svm_type eq 'nu_svc'){ @alpha = $self->solve_nu_svc($model); }
  elsif($model->svm_type eq 'one_class'){ @alpha = $self->solve_one_class($model); }
  elsif($model->svm_type eq 'epsilon_svr'){ @alpha = $self->solve_epsilon_svr($model); }
  elsif($model->svm_type eq 'nu_svr'){ @alpha = $self->solve_nu_svr($model); }
  if(!$self->{'quiet'}){ print STDERR "obj = ".$self->{'obj'}.", rho = ".$model->rho(0)."\n"; }
  # output SVs
  my $nSV = 0; my $nBSV = 0;
  for(my $row=0; $row < $self->{'problem'}{'count'}; $row++){
    if(abs($alpha[$row]) > 0){
      $nSV++;
      if($self->{'problem'}{'y'}[$row] > 0 and abs($alpha[$row]) >= $self->{'upper_bound_p'}){ $nBSV++; }
      elsif(abs($alpha[$row]) >= $self->{'upper_bound_n'}){ $nBSV++; }
    }
  }
  if(!$self->{'quiet'}){ print STDERR "nSV = ".$nSV.", nBSV = ".$nBSV."\n"; }
  return @alpha;
}

sub solve_epsilon_svr{
  my ($self,$model) = @_;
  my $l = $self->{'problem'}{'count'};
  my @alpha;
  my @alpha2; my @linear_term;
  my @y;
  for(my $i = 0; $i<$l; $i++){
    $alpha2[$i] = 0;
    $linear_term[$i] = $self->{'epsilon_svr'} - $self->{'problem'}{'y'}[$i];
    $y[$i] = 1;

    $alpha2[$i + $l] = 0;
    $linear_term[$i + $l] = $self->{'epsilon_svr'} + $self->{'problem'}{'y'}[$i];
    $y[$i + $l] = -1;
  }
  $self->{'Cn'} = $self->{'cost'}; $self->{'Cp'} = $self->{'cost'};
  $self->{'problem'}{'count'} = 2 * $l;
  @alpha2 = $self->solver(\@linear_term,\@y,\@alpha2,$model);
  $self->{'problem'}{'count'} = $l;
  my $sum_alpha = 0;
  for(my $i = 0;$i < $l;$i++){
    $alpha[$i] = $alpha2[$i] - $alpha2[$i + $l];
    $sum_alpha += abs($alpha[$i]);
  }
  if(!$self->{'quiet'}){
    print STDERR "sum_alpha $sum_alpha ".$self->{'cost'}." $l\n";
    print STDERR "nu = ".($sum_alpha/($self->{'cost'} * $l))."\n";
  }
  return @alpha;
}

sub solve_nu_svr{
  my ($self,$model) = @_;
  my $l = $self->{'problem'}{'count'};
#  $self->loger("solve_nu_svr");
  my $C = $self->{'cost'};
  my @alpha;
  my @alpha2; my @linear_term; my @y;
  my $sum = $C * $self->{'nu'} * $l / 2;
  for(my $i = 0; $i < $l; $i++){
    my $min = $C; if($sum < $min){ $min = $sum;}
    $alpha2[$i] = $min;
    $alpha2[$i + $l] = $min;
    $sum -= $alpha2[$i];
    $linear_term[$i] = - $self->{'problem'}{'y'}[$i];
    $y[$i] = 1;
    $linear_term[$i + $l] = $self->{'problem'}{'y'}[$i];
    $y[$i + $l] = -1;
  }
  $self->{'Cn'} = $C; $self->{'Cp'} = $C;
  $self->{'problem'}{'count'} = 2 * $l;
  @alpha2 = $self->solver(\@linear_term,\@y,\@alpha2,$model);
  $self->{'problem'}{'count'} = $l;
  if(!$self->{'quiet'}){ print STDERR "epsilon = ".(0 - $self->{'epsilon_svr'})."\n"; }
  for(my $i = 0; $i < $l; $i++){
    $alpha[$i] = $alpha2[$i] - $alpha2[$i + $l];
  }
  return @alpha;
}

sub solve_c_svc {
  my ($self,$model) = @_;
  my $l = $self->{'problem'}{'count'};
  my @alpha; my @y;
  my @minus_ones;
  for(my $i = 0; $i < $l; $i++){
    $alpha[$i] = 0;
    $minus_ones[$i] = -1;
    if($self->{'problem'}{'y'}[$i] > 0){ $y[$i] = 1; }
    else{ $y[$i] = -1; }
  }
  @alpha = $self->solver(\@minus_ones,\@y,\@alpha,$model);
  my $sum_alpha = 0;
  for(my $i = 0; $i < $l; $i++){ $sum_alpha += $alpha[$i]; }
  if($self->{'Cp'} == $self->{'Cn'} and !$self->{'quiet'}){
    print STDERR "nu = ".($sum_alpha/($self->{'Cp'} * $l))."\n";
  }
  for(my $i = 0; $i < $l; $i++){ $alpha[$i] *= $y[$i]; }
  return @alpha;
}

sub solve_nu_svc {
  my ($self,$model) = @_;
  my $l = $self->{'problem'}{'count'};
  my $nu = $self->{'nu'};
  my @y; my @alpha;
  for(my $i = 0; $i < $l; $i++){
    if($self->{'problem'}{'y'}[$i] > 0){ $y[$i] = 1; }
    else{ $y[$i] = -1; }
  }
  my $sum_pos = $nu * $l / 2; my $sum_neg = $nu * $l / 2;
  for(my $i = 0; $i < $l; $i++){
    if($y[$i] == 1){
      $alpha[$i] = 1; 
      if($sum_pos < $alpha[$i]){ $alpha[$i] = $sum_pos; }
      $sum_pos -= $alpha[$i];
    }
    else{
      $alpha[$i] = 1;
      if($sum_neg < $alpha[$i]){ $alpha[$i] = $sum_neg; }
      $sum_neg -= $alpha[$i];
    }
  }
  my @zeros;
  for(my $i = 0; $i < $l; $i++){ $zeros[$i] = 0; }
  $self->{'Cp'} = 1; $self->{'Cn'} = 1;
  @alpha = $self->solver(\@zeros,\@y,\@alpha,$model);
  my $r = $self->{'epsilon_svr'};
  if(!$self->{'quiet'}){ print STDERR "C = ".(1 / $r)."\n"; }
  for(my $i = 0; $i < $l; $i++){ $alpha[$i] *= $y[$i] / $r; }
  $self->{'rho'} /= $r;
  $self->{'obj'} /= ($r ** 2);
  $self->{'upper_bound_p'} = 1 / $r;
  $self->{'upper_bound_n'} = 1 / $r;
  return @alpha;
}

sub solve_one_class {
  my ($self,$model) = @_;
  my $l = $self->{'problem'}{'count'};
  my @zeros; my @ones;
  my @alpha;
  my $n = $self->{'nu'} * $l; # Num of alpha's at upper bound

  for(my $i = 0; $i < $n; $i++){ $alpha[$i] = 1; }
  if($n < $l){ $alpha[$n] = $self->{'nu'} * $l - $n; }
  for(my $i=$n+1; $i < $l; $i++){ $alpha[$i] = 0; }
  for(my $i=0;$i<$l;$i++){
    $zeros[$i] = 0;
    $ones[$i] = 1;
  }
  $self->{'Cp'} = 1;
  $self->{'Cn'} = 1;
  return $self->solver(\@zeros,\@ones,\@alpha,$model);
}

sub get_C {
  my ($self,$row,$y) = @_;
  if($$y[$row] > 0){ return $self->{'Cp'}; }
  else{ return $self->{'Cn'}; }
}

sub _initialize_gradient { # threaded gradient builder
  my ($self,$i_start,$i_end,$model,$y,$alpha,$alpha_status) = @_;
  my @G;
  my @G_bar;
  my $gamma = $model->gamma();
  my $coef0 = $model->coef0();
  my $degree = $model->degree();
  my $keySet = $self->{'keySet'};
  for(my $i = $i_start ; $i < $i_end; $i++){
    if($$alpha_status[$i] ne 'lower_bound'){
      my @Q_i = $self->get_Q($keySet,$i,$gamma,$coef0,$degree,$y);
      my $alpha_i = $$alpha[$i];
      for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){
        $G[$j] += $alpha_i * $Q_i[$j];
      }
      if($$alpha_status[$i] eq 'upper_bound'){
        for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){
          $G_bar[$j] += $self->get_C($i,$y) * $Q_i[$j];
        }
      }
    }
  }
  return (\@G,\@G_bar);
}

sub set_types{
  my ($self,$model) = @_;

  my $kernel_type = $model->kernel_type();
  my $svm_type = $model->svm_type();

  if($kernel_type eq 'linear'){ *_kernel = \&_kernel_linear; }
  elsif($kernel_type eq 'polynomial'){ *_kernel = \&_kernel_polynomial; }
  elsif($kernel_type eq 'rbf'){ *_kernel = \&_kernel_rbf; }
  elsif($kernel_type eq 'sigmoid'){ *_kernel = \&_kernel_sigmoid; }
  elsif($kernel_type eq 'precomputed'){ *_kernel = \&_kernel_precomputed; }
  else{ die "Could not identify kernel type $kernel_type\n"; }

  if($svm_type eq 'one_class'){ *get_Q = \&get_Q_one_class; }
  elsif($svm_type eq 'c_svc' or $svm_type eq 'nu_svc'){ *get_Q = \&get_Q_svc; }
  elsif($svm_type eq 'epsilon_svr' or $svm_type eq 'nu_svr'){ *get_Q = \&get_Q_svr; }
  else{ die "Could not identify svm type $svm_type\n"; }
}

sub solver {
  my ($self,$p,$y,$alpha_ret,$model) = @_;
#  $self->loger("Run Solver");
  my @QD = $self->get_QD($model);
  my @alpha = @$alpha_ret;
  # initialize alpha_status
#  $self->loger("initialize alpha_status");
  my @alpha_status;
  for(my $row=0; $row < $self->{'problem'}{'count'}; $row++){
    if($alpha[$row] >= $self->get_C($row,$y)){ $alpha_status[$row] = 'upper_bound'; }
    elsif($alpha[$row] <= 0){ $alpha_status[$row] = 'lower_bound'; }
    else{ $alpha_status[$row] = 'free'; }
  }
  # initialize active set (for shrinking)
#  $self->loger("initialize active set (for shrinking)");
  my @active_set;
  for(my $row=0; $row < $self->{'problem'}{'count'}; $row++){
    $active_set[$row] = $row;
  }
  my $active_size = $self->{'problem'}{'count'};
  # initialize gradient
#  $self->loger("initialize gradient");
  my @G = (0) x $active_size; my @G_bar = (0) x $active_size;
  for(my $i=0; $i < $self->{'problem'}{'count'}; $i++){
    $G[$i] = $$p[$i];
    $G_bar[$i] = 0;
  }

  if($self->{'threads'}){
    my @threads;
    my $inc = int($self->{'problem'}{'count'} / $self->{'maxthreads'});
    for(my $i=0; $i < $self->{'problem'}{'count'}; $i += $inc){
      my $end = $i + $inc;
      if($end > $self->{'problem'}{'count'}){ $end = $self->{'problem'}{'count'}; }
      push @threads, threads->create('_initialize_gradient',
        $self,$i,$end,$model,$y,\@alpha,\@alpha_status);
    }
    for(my $i=0; $i < $self->{'problem'}{'count'}; $i += $inc){
      my $end = $i + $inc;
      if($end > $self->{'problem'}{'count'}){ $end = $self->{'problem'}{'count'}; }
      my $thread = shift @threads;
      my ($G_ret, $G_bar_ret) = $thread->join();
      for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){
        $G[$j] += $$G_ret[$j];
        $G_bar[$j] += $$G_bar_ret[$j];
      }
    }
  }
  else{
    my $gamma = $model->gamma();
    my $coef0 = $model->coef0();
    my $degree = $model->degree();
    my $keySet = $self->{'keySet'};
    for(my $i=0; $i < $self->{'problem'}{'count'}; $i++){
      if($alpha_status[$i] ne 'lower_bound'){
        my @Q_i = $self->get_Q($keySet,$i,$gamma,$coef0,$degree,$y);
        my $alpha_i = $alpha[$i];
        for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){
          $G[$j] += $alpha_i * $Q_i[$j];
        }
        if($alpha_status[$i] eq 'upper_bound'){
          for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){
            $G_bar[$j] += $self->get_C($i,$y) * $Q_i[$j];
          }
        }
      }
    }
  }
  
  # optimization step
#  $self->loger("optimization step");
  my $iter = 0;
  my $max_iter = 10000000;
  if($self->{'problem'}{'count'} * 100 > $max_iter){ $max_iter = $self->{'problem'}{'count'} * 100; }
  my $counter = 1000;
  if($counter > $self->{'problem'}{'count'}){ $counter = $self->{'problem'}{'count'}; }
  $counter++;
  my $gamma = $model->gamma();
  my $coef0 = $model->coef0();
  my $degree = $model->degree();
  my $keySet = $self->{'keySet'};
  while($iter < $max_iter){
    # show progress and do shrinking
    $counter--;
    if($counter == 0){
      $counter = 1000;
      if($counter > $self->{'problem'}{'count'}){ $counter = $self->{'problem'}{'count'}; }
      if($self->{'shrinking'}){ $self->do_shrinking(); }
      if(!$self->{'quiet'}){ print STDERR '.'; }
    }
    my $i; my $j;
#    $self->loger("select_working_set");
    if($self->select_working_set(\$i,\$j,$y,\@QD,\@G,$active_size,\@alpha_status,$model) ne 0){
      # reconstruct the whole gradient
#      $self->loger("reconstruct the whole gradient");
      $self->reconstruct_gradient($y,\@alpha,\@G,\@G_bar,$p,$active_size,$model);
      # reset active set size and check
      $active_size = $self->{'problem'}{'count'};
      if(!$self->{'quiet'}){ print STDERR '*'; }
#      $self->loger("select_working_set");
      if($self->select_working_set(\$i,\$j,$y,\@QD,\@G,$active_size,\@alpha_status,$model) ne 0){ last; }
      else{ $counter = 1; } # do shrinking next iteration
    }
    $iter++;
    # update alpha[i] and alpha[j], handle bounds carefully
#    $self->loger("update alpha[i] and alpha[j], handle bounds carefully");
    my @Q_i = $self->get_Q($keySet,$i,$gamma,$coef0,$degree,$y);
    my @Q_j = $self->get_Q($keySet,$j,$gamma,$coef0,$degree,$y);
    
    my $C_i = $self->get_C($i,$y);
    my $C_j = $self->get_C($j,$y);

    my $old_alpha_i = $alpha[$i];
    my $old_alpha_j = $alpha[$j];

    if($$y[$i] ne $$y[$j]){
      my $quad_coef = $QD[$i]+$QD[$j]+ 2 * $Q_i[$j];
      if($quad_coef <= 0){ $quad_coef = 1e-12; }
      my $delta = (0 - $G[$i] - $G[$j]) / $quad_coef;
      my $diff = $alpha[$i] - $alpha[$j];
      $alpha[$i] += $delta;
      $alpha[$j] += $delta;
      if($diff > 0 and $alpha[$j] < 0){
        $alpha[$j] = 0;
        $alpha[$i] = $diff;
      }
      elsif($alpha[$i] < 0){
        $alpha[$i] = 0;
        $alpha[$j] = 0 - $diff;
      }
      if(($diff > $C_i - $C_j) and ($alpha[$i] > $C_i)){
        $alpha[$i] = $C_i;
        $alpha[$j] = $C_i - $diff;
      }
      elsif($alpha[$j] > $C_j){
        $alpha[$j] = $C_j;
        $alpha[$i] = $C_j + $diff;
      }
    }
    else{
      my $quad_coef = $QD[$i] + $QD[$j] - 2 * $Q_i[$j];
      if($quad_coef <= 0){ $quad_coef = 1e-12; }
      my $delta = ( $G[$i] - $G[$j] ) / $quad_coef;
      my $sum = $alpha[$i] + $alpha[$j];
      $alpha[$i] -= $delta;
      $alpha[$j] += $delta;
      if($sum > $C_i and $alpha[$i] > $C_i){
        $alpha[$i] = $C_i;
        $alpha[$j] = $sum - $C_i;
      }
      elsif($alpha[$j] < 0){
        $alpha[$j] = 0;
        $alpha[$i] = $sum;
      }
      if($sum > $C_j and $alpha[$j] > $C_j){
        $alpha[$j] = $C_j;
        $alpha[$i] = $sum - $C_j;
      }
      elsif($alpha[$i] < 0){
        $alpha[$i] = 0;
        $alpha[$j] = $sum;
      }
    }
    # update G
#    $self->loger("update G");
    my $delta_alpha_i = $alpha[$i] - $old_alpha_i;
    my $delta_alpha_j = $alpha[$j] - $old_alpha_j;
    for(my $k = 0;$k < $active_size;$k++){
      $G[$k] += $Q_i[$k] * $delta_alpha_i + $Q_j[$k] * $delta_alpha_j;
    }
#    $self->loger("update alpha_status and G_bar");
    # update alpha_status and G_bar
    {
      my $ui = $alpha_status[$i];
      my $uj = $alpha_status[$j];
      {
        my $C = $self->{'Cn'};
        if($$y[$i] > 0){ $C = $self->{'Cp'}; }
        if($alpha[$i] >= $C){ $alpha_status[$i] = 'upper_bound'; }
        elsif($alpha[$i] <= 0){ $alpha_status[$i] = 'lower_bound'; }
        else{ $alpha_status[$i] = 'free'; }
      }
      {
        my $C = $self->{'Cn'};
        if($$y[$j] > 0){ $C = $self->{'Cp'}; }
        if($alpha[$j] >= $C){ $alpha_status[$j] = 'upper_bound'; }
        elsif($alpha[$j] <= 0){ $alpha_status[$j] = 'lower_bound'; }
        else{ $alpha_status[$j] = 'free'; }
      }
      if($alpha_status[$i] eq 'upper_bound' and $ui eq $alpha_status[$i]){
        @Q_i = $self->get_Q($keySet,$i,$gamma,$coef0,$degree,$y);
        if($ui eq 'upper_bound'){
          for(my $k = 0;$k<$self->{'problem'}{'count'};$k++){ $G_bar[$k] -= $C_i * $Q_i[$k]; }
        }
        else{
          for(my $k = 0;$k<$self->{'problem'}{'count'};$k++){ $G_bar[$k] += $C_i * $Q_i[$k]; }
        }
      }
      if($alpha_status[$j] eq 'upper_bound' and $uj eq $alpha_status[$j]){
        @Q_j = $self->get_Q($keySet,$j,$gamma,$coef0,$degree,$y);
        if($uj eq 'upper_bound'){
          for(my $k = 0;$k<$self->{'problem'}{'count'};$k++){ $G_bar[$k] -= $C_j * $Q_j[$k]; }
        }
        else{
          for(my $k = 0;$k<$self->{'problem'}{'count'};$k++){ $G_bar[$k] += $C_j * $Q_j[$k]; }
        }
      }
    }
  }
  if($iter >= $max_iter){
    if($active_size < $self->{'problem'}{'count'}){
      # reconstruct the whole gradient to calculate objective value
      $self->reconstruct_gradient($y,\@alpha,\@G,\@G_bar,$p,$active_size,$model);
      $active_size = $self->{'problem'}{'count'};
      if(!$self->{'quiet'}){ print STDERR '*'; }
    }
    if(!$self->{'quiet'}){ print STDERR "\nWARNING: reaching max number of iterations"; }
  }
  # calculate rho
#  $self->loger("calculate rho");
  $model->set_rho(0,$self->calculate_rho($y,\@G,\@alpha_status,$active_size,$model));
  # calculate objective value
#  $self->loger("calculate objective value");
  {
    my $v = 0;
    for(my $i = 0;$i < $self->{'problem'}{'count'};$i++){
      $v += $alpha[$i] * ($G[$i] + $$p[$i]);
    }
    $self->{'obj'} = $v / 2;
  }
  # put back the solution
#  $self->loger("put back the solution");
  {
    for(my $i = 0;$i < $self->{'problem'}{'count'};$i++){
      $$alpha_ret[$active_set[$i]] = $alpha[$i];
    }
  }
  $self->{'upper_bound_p'} = $self->{'Cp'};
  $self->{'upper_bound_n'} = $self->{'Cn'};
  if(!$self->{'quiet'}){ print STDERR "\noptimization finished, #iter = $iter\n"; }
  return @alpha;
}

sub calculate_rho {
  my ($self,$y,$G,$alpha_status,$active_size,$model) = @_;
  if($model->svm_type eq 'nu_svc' or $model->svm_type eq 'nu_svr'){
    return $self->calculate_rho_nu_class($y,$G,$alpha_status,$active_size);
  }
  else{
    return $self->calculate_rho_one_class($y,$G,$alpha_status,$active_size);
  }
}

sub calculate_rho_nu_class {
  my ($self,$y,$G,$alpha_status,$active_size) = @_;
  my $rho;
  my $nr_free1 = 0; my $nr_free2 = 0;
  my $ub1 = 9**9**9; my $ub2 = 9**9**9;
  my $lb1 = -9**9**9; my $lb2 = -9**9**9;
  my $sum_free1 = 0; my $sum_free2 = 0;
  for(my $i = 0; $i < $active_size; $i++){
    if($$y[$i] == 1){
      if($$alpha_status[$i] eq 'upper_bound'){
        if($lb1 < $$G[$i]){ $lb1 = $$G[$i]; }
      }
      elsif($$alpha_status[$i] eq 'lower_bound'){
        if($ub1 > $$G[$i]){ $ub1 = $$G[$i]; }
      }
      else{
        $nr_free1++;
        $sum_free1 += $$G[$i];
      }
    }
    else{
      if($$alpha_status[$i] eq 'upper_bound'){
        if($lb2 < $$G[$i]){ $lb2 = $$G[$i]; }
      }
      elsif($$alpha_status[$i] eq 'lower_bound'){
        if($ub2 > $$G[$i]){ $ub2 = $$G[$i]; }
      }
      else{
        $nr_free2++;
        $sum_free2 += $$G[$i];
      }
    }
  }
  my $rho1;
  if($nr_free1 > 0){ $rho1 = $sum_free1 / $nr_free1; }
  else{ $rho1 = ($ub1 + $lb1) / 2; }
  my $rho2;
  if($nr_free2 > 0){ $rho2 = $sum_free2 / $nr_free2; }
  else{ $rho2 = ($ub2 + $lb2) / 2; }
  $self->{'epsilon_svr'} = (($rho1 + $rho2) / 2);
  return (($rho1 - $rho2) / 2);
}

sub calculate_rho_one_class {
  my ($self,$y,$G,$alpha_status,$active_size) = @_;
  my $rho;
  my $nr_free = 0;
  my $ub = 9**9**9; my $lb = -9**9**9; my $sum_free = 0;
  for(my $i = 0; $i < $active_size; $i++){
    my $yG = $$y[$i] * $$G[$i];
    if($$alpha_status[$i] eq 'upper_bound'){
      if($$y[$i] == -1){ if($ub > $yG){$ub = $yG;} }
      else{ if($lb < $yG){$lb = $yG;} }
    }
    elsif($$alpha_status[$i] eq 'lower_bound'){
      if($$y[$i] == 1){ if($ub > $yG){$ub = $yG;} }
      else{ if($lb < $yG){$lb = $yG;} }
    }
    else{
      $nr_free++;
      $sum_free += $yG;
    }
  }
  if($nr_free > 0){ $rho = $sum_free / $nr_free; }
  else{ $rho = ($ub + $lb) / 2; }
  $self->{'rho'} = $rho;
  return $rho;
}

sub reconstruct_gradient{
  my ($self,$y,$alpha,$G,$G_bar,$p,$active_size,$model) = @_;
  # reconstruct inactive elements of G from G_bar and free variables
  if($active_size == $self->{'problem'}{'count'}){ return; }
  my $nr_free = 0;
  for(my $j = $active_size;$j < $self->{'problem'}{'count'};$j++){
    $$G[$j] = $$G_bar[$j] + $$p[$j];
  }
  for(my $j = 0;$j < $active_size; $j++){
    if($self->is_free($j)){ $nr_free++; }
  }
  if(2 * $nr_free < $active_size and !$self->{'quiet'}){ print STDERR "\nWARNING: using -h 0 may be faster\n"; }
  my $gamma = $model->gamma();
  my $coef0 = $model->coef0();
  my $degree = $model->degree();
  my $keySet = $self->{'keySet'};
  if($nr_free * $self->{'problem'}{'count'} > 2 * $active_size * ($self->{'problem'}{'count'} - $active_size)){
    for(my $i = $active_size; $i < $self->{'problem'}{'count'}; $i++){
      my @Q_i = $self->get_Q($keySet,$i,$gamma,$coef0,$degree,$y);
      for(my $j = 0; $j < $active_size; $j++){
        if($self->is_free($j)){ $$G[$i] += $$alpha[$j] * $self->get_Q($keySet,$j,$gamma,$coef0,$degree,$y); }
      }
    }
  }
  else{
    for(my $i = 0;$i < $active_size; $i++){
      if($self->is_free($i)){
        my $Q_i = $self->get_Q($keySet,$i,$gamma,$coef0,$degree,$y);
        my $alpha_i = $$alpha[$i];
        for(my $j = $active_size; $j < $self->{'problem'}{'count'}; $j++){ $$G[$j] += $alpha_i * $self->get_Q($keySet,$j,$gamma,$coef0,$degree,$y); }
      }
    }
  }
}

sub do_shrinking{ #TODO Build
}

sub select_working_set {
  # return i,j such that
  # i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
  # j: minimizes the decrease of obj value
  #    (if quadratic coefficeint <= 0, replace it with tau)
  #    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
  my ($self,$i,$j,$y,$QD,$G,$active_size,$alpha_status,$model) = @_;
  if($model->svm_type eq 'nu_svc' or $model->svm_type eq 'nu_svr'){
    return $self->select_working_set_nu_class(
      $i,$j,$y,$QD,$G,$active_size,$alpha_status,$model);
  }
  else{
    return $self->select_working_set_one_class(
      $i,$j,$y,$QD,$G,$active_size,$alpha_status,$model);
  }
}

sub select_working_set_nu_class {
  my ($self,$i_ref,$j_ref,$y,$QD,$G,$active_size,$alpha_status,$model) = @_;
  my $Gmaxp = -9**9**9; # Negative Infinite
  my $Gmaxp2 = -9**9**9; # Negative Infinite
  my $Gmaxp_idx = -1;
  my $Gmaxn = -9**9**9; # Negative Infinite
  my $Gmaxn2 = -9**9**9; # Negative Infinite
  my $Gmaxn_idx = -1;
  my $Gmin_idx = -1;
  my $obj_diff_min = 9**9**9; # Positive Infinite
  my $gamma = $model->gamma();
  my $coef0 = $model->coef0();
  my $degree = $model->degree();
  my $keySet = $self->{'keySet'};
  for(my $t = 0;$t < $active_size;$t++){
    if($$y[$t] == 1){
      if($$alpha_status[$t] ne 'upper_bound'){
        if(0 - $$G[$t] >= $Gmaxp){
          $Gmaxp = 0 - $$G[$t];
          $Gmaxp_idx = $t;
        }
      }
    }
    else{
      if($$alpha_status[$t] ne 'lower_bound'){
        if($$G[$t] >= $Gmaxn){
          $Gmaxn = $$G[$t];
          $Gmaxn_idx = $t;
        }
      }
    }
  }

  my $ip = $Gmaxp_idx;
  my $in = $Gmaxn_idx;
  my @Q_ip; my @Q_in;
  if($ip != -1){ @Q_ip = $self->get_Q($keySet,$ip,$gamma,$coef0,$degree,$y); }
  if($in != -1){ @Q_in = $self->get_Q($keySet,$ip,$gamma,$coef0,$degree,$y); }
  for(my $j = 0; $j < $active_size; $j++){
    if($$y[$j] == 1){
      if($$alpha_status[$j] ne 'lower_bound'){
        my $grad_diff = $Gmaxp + $$G[$j];
        if($$G[$j] >= $Gmaxp2){ $Gmaxp2 = $$G[$j]; }
        if($grad_diff > 0){
          my $obj_diff;
          my $quad_coef = $$QD[$ip] + $$QD[$j] - 2 * $Q_ip[$j];
          if($quad_coef > 0){ $obj_diff = 0 - ($grad_diff ** 2) / $quad_coef; }
          else{ $obj_diff = 0 - ($grad_diff ** 2) / 1e-12; }
          if($obj_diff <= $obj_diff_min){
            $Gmin_idx = $j;
            $obj_diff_min = $obj_diff;
          }
        }
      }
    }
    else{
      if($$alpha_status[$j] ne 'upper_bound'){
        my $grad_diff= $Gmaxn - $$G[$j];
        if(0 - $$G[$j] >= $Gmaxn2){ $Gmaxn2 = 0 - $$G[$j]; }
        if($grad_diff > 0){
          my $obj_diff;
          my $quad_coef = $$QD[$in] + $$QD[$j] - 2 * $Q_in[$j];
          if($quad_coef > 0){ $obj_diff = 0 - ($grad_diff ** 2) / $quad_coef; }
          else{ $obj_diff = 0 - ($grad_diff ** 2) / 1e-12; }
          if($obj_diff <= $obj_diff_min){
            $Gmin_idx = $j;
            $obj_diff_min = $obj_diff;
          }
        }
      }
    }
  }
  my $maxGmax = $Gmaxp + $Gmaxp2;
  if($Gmaxn + $Gmaxn2 > $maxGmax){ $maxGmax = $Gmaxn + $Gmaxn2; }
  if($maxGmax < $self->{'epsilon_trm'}){ return 1; }
  if($$y[$Gmin_idx] == 1){ $$i_ref = $Gmaxp_idx; }
  else{ $$i_ref = $Gmaxn_idx; }
  $$j_ref = $Gmin_idx;
  return 0;
}

sub select_working_set_one_class {
  my ($self,$i_ref,$j_ref,$y,$QD,$G,$active_size,$alpha_status,$model) = @_;
  my $Gmax = -9**9**9; # Negative Infinite
  my $Gmax2 = -9**9**9; # Negative Infinite
  my $Gmax_idx = -1;
  my $Gmin_idx = -1;
  my $obj_diff_min = 9**9**9; # Positive Infinite
  my $gamma = $model->gamma();
  my $coef0 = $model->coef0();
  my $degree = $model->degree();
  my $keySet = $self->{'keySet'};
  for(my $t = 0;$t < $active_size;$t++){
    if($$y[$t] == 1){
      if($$alpha_status[$t] ne 'upper_bound'){
        if(-$$G[$t] >= $Gmax){
          $Gmax = -$$G[$t];
          $Gmax_idx = $t;
        }
      }
    }
    else{
      if($$alpha_status[$t] ne 'lower_bound'){
        if($$G[$t] >= $Gmax){
          $Gmax = $$G[$t];
          $Gmax_idx = $t;
        }
      }
    }
  }

  my $i = $Gmax_idx;
  my @Q_i;
  if($i ne -1){ @Q_i = $self->get_Q($keySet,$i,$gamma,$coef0,$degree,$y); }
  for(my $j = 0; $j < $active_size; $j++){
    if($$y[$j] == 1){
      if($$alpha_status[$j] ne 'lower_bound'){
        my $grad_diff = $Gmax + $$G[$j];
        if($$G[$j] >= $Gmax2){ $Gmax2 = $$G[$j]; }
        if($grad_diff > 0){
          my $obj_diff;
          my $quad_coef = $$QD[$i] + $$QD[$j] - 2 * $$y[$i] * $Q_i[$j];
          if($quad_coef > 0){ $obj_diff = -($grad_diff ** 2) / $quad_coef; }
          else{ $obj_diff = -($grad_diff ** 2) / 1e-12; }
          if($obj_diff <= $obj_diff_min){
            $Gmin_idx = $j;
            $obj_diff_min = $obj_diff;
          }
        }
      }
    }
    else{
      if($$alpha_status[$j] ne 'upper_bound'){
        my $grad_diff= $Gmax - $$G[$j];
        if(-$$G[$j] >= $Gmax2){ $Gmax2 = -$$G[$j]; }
        if($grad_diff > 0){
          my $obj_diff;
          my $quad_coef = $$QD[$i] + $$QD[$j] + 2 * $$y[$i] * $Q_i[$j];
          if($quad_coef > 0){ $obj_diff = -($grad_diff ** 2) / $quad_coef; }
          else{ $obj_diff = -($grad_diff ** 2) / 1e-12; }
          if($obj_diff <= $obj_diff_min){
            $Gmin_idx = $j;
            $obj_diff_min = $obj_diff;
          }
        }
      }
    }
  }
  if($Gmax + $Gmax2 < $self->{'epsilon_trm'}){ return 1; }
  $$i_ref = $Gmax_idx;
  $$j_ref = $Gmin_idx;
  return 0;
}

sub get_QD{
  my ($self,$model) = @_;
  my @QD;
  my $gamma = $model->gamma();
  my $coef0 = $model->coef0();
  my $degree = $model->degree();
  my $keySet = $self->{'keySet'};
  if($model->svm_type eq 'one_class' or 
    $model->svm_type eq 'c_svc' or 
    $model->svm_type eq 'nu_svc'){
    for(my $row=0; $row < $self->{'problem'}{'count'}; $row++){
      my $xSet = $self->{'problem'}{'x'}[$row];
      $QD[$row] = _kernel($keySet,$xSet,$xSet,$gamma,$coef0,$degree);
    }
  }
  elsif($model->svm_type eq 'epsilon_svr' or $model->svm_type eq 'nu_svr'){
    my $l = $self->{'problem'}{'count'} / 2;
    for(my $row=0; $row < $l; $row++){
      my $xSet = $self->{'problem'}{'x'}[$row];
      $QD[$row] = _kernel($keySet,$xSet,$xSet,$gamma,$coef0,$degree);
      $QD[$row + $l] = $QD[$row];
    }
  }
  return @QD;
}

sub get_Q_one_class{
  my ($self,$keySet,$i,$gamma,$coef0,$degree) = @_;
  if(defined($self->{'Q'}{$i})){ return @{ $self->{'Q'}{$i} }; }
  my @Q;
  my $xSet = $self->{'problem'}{'x'}[$i];
  for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){
    $Q[$j] = _kernel($keySet,$xSet,$self->{'problem'}{'x'}[$j],$gamma,$coef0,$degree);
  }
  $self->{'Q'}{$i} = \@Q;
  return @Q;
}

sub get_Q_svc{
  my ($self,$keySet,$i,$gamma,$coef0,$degree,$y) = @_;
  if(defined($self->{'Q'}{$i})){ return @{ $self->{'Q'}{$i} }; }
  my @Q;
  my $xSet = $self->{'problem'}{'x'}[$i];
  for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){
    $Q[$j] = $$y[$i] * $$y[$j] * 
    _kernel($keySet,$xSet,$self->{'problem'}{'x'}[$j],$gamma,$coef0,$degree);
  }
  $self->{'Q'}{$i} = \@Q;
  return @Q;
}

sub get_Q_svr{
  my ($self,$keySet,$i,$gamma,$coef0,$degree) = @_;
  my @Q;
  my $l = $self->{'problem'}{'count'} / 2;
  my @sign; my @index;
  if( defined($self->{'get_Q_sign'}) and defined($self->{'get_Q_index'}) ){
    @sign = @{ $self->{'get_Q_sign'} };
    @index = @{ $self->{'get_Q_index'} };
  }
  else{
    for(my $k=0; $k < $l; $k++){
      $sign[$k] = 1; $sign[$k+$l] = -1;
      $index[$k] = $k; $index[$k+$l] = $k;
    }
    $self->{'get_Q_sign'} = \@sign;
    $self->{'get_Q_index'} = \@index;
  }
  my $real_i = $index[$i];
  my $xSet = $self->{'problem'}{'x'}[$real_i];
  my @Qorig;
  for(my $j=0; $j < $l; $j++){
    $Qorig[$j] = _kernel($keySet,$xSet,$self->{'problem'}{'x'}[$j],$gamma,$coef0,$degree);
  }
  my $si = $sign[$i];
  for(my $j=0; $j < $self->{'problem'}{'count'}; $j++){ # reorder and copy
    $Q[$j] = $si * $sign[$j] * $Qorig[$index[$j]];
  }
  return @Q;
}

sub svr_probability{ # Return parameter of a Laplace distribution
  my ($self,$model) = @_;
  my @ymv; my $mae = 0;
  my $newsvm = Algorithm::ML::SVM->new();
  $self->param_copy($newsvm);
  $newsvm->{'probability'} = 0;
  $newsvm->{'n_fold'} = 5;
  $newsvm->cross_validation($model,\@ymv);
  my $l = $self->{'problem'}{'count'};
  for(my $i=0;$i<$l;$i++){
    $ymv[$i] = $self->{'problem'}{'y'}[$i] - $ymv[$i];
    $mae += abs($ymv[$i]);
  }
  $mae /= $l;
  my $std = sqrt(2 * ($mae ** 2));
  my $count = 0;
  $mae = 0;
  for(my $i=0;$i<$l;$i++){
    if(abs($ymv[$i]) > 5 * $std){ $count++; }
    else{ $mae += abs($ymv[$i]); }
  }
  $mae /= ($l - $count);
  if(!$self->{'quiet'}){
    print STDERR "Prob. model for test data: target value = predicted value + z,\n".
    "z: Laplace distribution e^(-|z|/sigma)/(2sigma),sigma= $mae\n";
  }
  return $mae;
}

sub cross_validation { # Statified cross validation
  my ($self,$model,$target) = @_;
  if($self->{'threads'}){ return $self->cross_validation_thr($model,$target); }
  else{ return $self->cross_validation_lin($model,$target); }
}

sub cross_validation_lin { # linear mode
  my ($self,$model,$target) = @_;

  my @fold_start; my @perm;
  my $nr_class;
  my $n_fold = $self->{'n_fold'};
  my $l = $self->{'problem'}{'count'};
  # stratified cv may not give leave-one-out rate
  # Each class to l folds -> some folds may have zero elements
  if(($model->svm_type eq 'c_svc' or $model->svm_type eq 'nu_svc') and
    $n_fold < $l){
    my @start; my @label; my @count;
    $self->group_classes(\$nr_class,\@label,\@start,\@count,\@perm);
    
    # random shuffle and then data grouped by fold using the array perm
    my @fold_count;
    my @index;
    for(my $i=0;$i<$l;$i++){ $index[$i] = $perm[$i]; }
    for(my $c=0; $c<$nr_class; $c++){
      for(my $i=0;$i<$count[$c];$i++){
        my $j = $i + int(rand($count[$c] - $i));
        ($index[$start[$c] + $i],$index[$start[$c] + $j]) = 
        ($index[$start[$c] + $j], $index[$start[$c] + $i]);
      }
    }
    for(my $i=0;$i<$n_fold;$i++){
      $fold_count[$i] = 0;
      for(my $c=0; $c<$nr_class; $c++){
        $fold_count[$i] += ($i + 1) * $count[$c] / $n_fold - $i * $count[$c] / $n_fold;
      }
    }
    $fold_start[0] = 0;
    for(my $i=1;$i<=$n_fold;$i++){
      $fold_start[$i] = $fold_start[$i - 1] + $fold_count[$i - 1];
    }
    for(my $c=0; $c<$nr_class; $c++){
      for(my $i=0;$i<$n_fold;$i++){
        my $begin = int($start[$c] + $i * $count[$c] / $n_fold);
        my $end = int($start[$c] + ($i + 1) * $count[$c] / $n_fold);
        for(my $j=$begin;$j<$end;$j++){
          $perm[$fold_start[$i]] = $index[$j];
          $fold_start[$i]++;
        }
      }
    }
    $fold_start[0]=0;
    for(my $i=1;$i<=$n_fold;$i++){
      $fold_start[$i] = $fold_start[$i - 1] + $fold_count[$i - 1];
    }
  }
  else{
    for(my $i=0;$i<$l;$i++){ $perm[$i]=$i; }
    for(my $i=0;$i<$l;$i++){
      my $j = $i + int(rand($l-$i));
      ($perm[$j],$perm[$i]) = ($perm[$i],$perm[$j]);
    }
    for(my $i=0;$i<=$n_fold;$i++){ $fold_start[$i]=$i * $l / $n_fold; }
  }
  for(my $i=0;$i<$n_fold;$i++){
    my $begin = $fold_start[$i];
    my $end = $fold_start[$i + 1];
    my $sub_prob = Algorithm::ML::SVM->new();

    $sub_prob->{'problem'}{'count'} = $l - ($end - $begin);

    my $k=0;
    for(my $j=0;$j<$begin;$j++){
      $sub_prob->{'problem'}{'x'}[$k] = $self->{'problem'}{'x'}[$perm[$j]];
      $sub_prob->{'problem'}{'y'}[$k] = $self->{'problem'}{'y'}[$perm[$j]];
      $k++;
    }
    for(my $j=$end;$j<$l;$j++){
      $sub_prob->{'problem'}{'x'}[$k] = $self->{'problem'}{'x'}[$perm[$j]];
      $sub_prob->{'problem'}{'y'}[$k] = $self->{'problem'}{'y'}[$perm[$j]];
      $k++;
    }
    $self->param_copy($sub_prob);
    my $sub_model = Algorithm::ML::SVM::Model->new();
    $model->param_copy($sub_model);
    $sub_prob->train($sub_model);
    if($self->{'probability'} and ($model->svm_type eq 'c_svc' or $model->svm_type eq 'nu_svc')) {
      my @prob_estimates;
      for(my $j=$begin;$j<$end;$j++){
        $$target[$perm[$j]] = $self->predict_probability(
          $sub_model,$self->{'problem'}{'x'}[$perm[$j]],\@prob_estimates);
      }
    }
    else{
      for(my $j=$begin;$j<$end;$j++){
        my $xSet = $self->{'problem'}{'x'}[$perm[$j]];
        my @dec_values;
        $$target[$perm[$j]] = $self->predict($sub_model,$xSet,\@dec_values);
      }
    }
  }
}

sub cross_validation_thr { # Threaded Mode
  my ($self,$model,$target) = @_;

  my @fold_start; my @perm;
  my $nr_class;
  my $n_fold = $self->{'n_fold'};
  my $l = $self->{'problem'}{'count'};
  # stratified cv may not give leave-one-out rate
  # Each class to l folds -> some folds may have zero elements
  if(($model->svm_type eq 'c_svc' or $model->svm_type eq 'nu_svc') and
    $n_fold < $l){
    my @start; my @label; my @count;
    $self->group_classes(\$nr_class,\@label,\@start,\@count,\@perm);
    
    # random shuffle and then data grouped by fold using the array perm
    my @fold_count;
    my @index;
    for(my $i=0;$i<$l;$i++){ $index[$i] = $perm[$i]; }
    for(my $c=0; $c<$nr_class; $c++){
      for(my $i=0;$i<$count[$c];$i++){
        my $j = $i + int(rand($count[$c] - $i));
        ($index[$start[$c] + $i],$index[$start[$c] + $j]) = 
        ($index[$start[$c] + $j], $index[$start[$c] + $i]);
      }
    }
    for(my $i=0;$i<$n_fold;$i++){
      $fold_count[$i] = 0;
      for(my $c=0; $c<$nr_class; $c++){
        $fold_count[$i] += ($i + 1) * $count[$c] / $n_fold - $i * $count[$c] / $n_fold;
      }
    }
    $fold_start[0] = 0;
    for(my $i=1;$i<=$n_fold;$i++){
      $fold_start[$i] = $fold_start[$i - 1] + $fold_count[$i - 1];
    }
    for(my $c=0; $c<$nr_class; $c++){
      for(my $i=0;$i<$n_fold;$i++){
        my $begin = int($start[$c] + $i * $count[$c] / $n_fold);
        my $end = int($start[$c] + ($i + 1) * $count[$c] / $n_fold);
        for(my $j=$begin;$j<$end;$j++){
          $perm[$fold_start[$i]] = $index[$j];
          $fold_start[$i]++;
        }
      }
    }
    $fold_start[0]=0;
    for(my $i=1;$i<=$n_fold;$i++){
      $fold_start[$i] = $fold_start[$i - 1] + $fold_count[$i - 1];
    }
  }
  else{
    for(my $i=0;$i<$l;$i++){ $perm[$i]=$i; }
    for(my $i=0;$i<$l;$i++){
      my $j = $i + int(rand($l-$i));
      ($perm[$j],$perm[$i]) = ($perm[$i],$perm[$j]);
    }
    for(my $i=0;$i<=$n_fold;$i++){ $fold_start[$i]=$i * $l / $n_fold; }
  }
  my @threads;
  for(my $i=0;$i<$n_fold;$i++){
    my $begin = $fold_start[$i];
    my $end = $fold_start[$i + 1];
    push @threads, threads->create('_cross_validation_solve',$self,$model,$begin,$end,\@perm);
  }
  for(my $i=0;$i<$n_fold;$i++){
    my $begin = $fold_start[$i];
    my $end = $fold_start[$i + 1];
    my $thread = shift @threads;
    my ($target_calculated,$message) = $thread->join();
    for(my $j=$begin;$j<$end;$j++){
      $$target[$perm[$j]] = $$target_calculated[$perm[$j]];
    }
  }
}

sub _cross_validation_solve {
  my ($self,$model,$begin,$end,$perm) = @_;
  my @target;
  my $message;
  my $l = $self->{'problem'}{'count'};

  my $sub_prob = Algorithm::ML::SVM->new();
  $sub_prob->{'problem'}{'count'} = $l - ($end - $begin);

  my $k=0;
  for(my $j=0;$j<$begin;$j++){
    $sub_prob->{'problem'}{'x'}[$k] = $self->{'problem'}{'x'}[$$perm[$j]];
    $sub_prob->{'problem'}{'y'}[$k] = $self->{'problem'}{'y'}[$$perm[$j]];
    $k++;
  }
  for(my $j=$end;$j<$l;$j++){
    $sub_prob->{'problem'}{'x'}[$k] = $self->{'problem'}{'x'}[$$perm[$j]];
    $sub_prob->{'problem'}{'y'}[$k] = $self->{'problem'}{'y'}[$$perm[$j]];
    $k++;
  }
  $self->param_copy($sub_prob);
  my $sub_model = Algorithm::ML::SVM::Model->new();
  $model->param_copy($sub_model);
  $sub_prob->train($sub_model);
  if($self->{'probability'} and ($model->svm_type eq 'c_svc' or $model->svm_type eq 'nu_svc')) {
    my @prob_estimates;
    for(my $j=$begin;$j<$end;$j++){
      $target[$$perm[$j]] = $self->predict_probability(
        $sub_model,$self->{'problem'}{'x'}[$$perm[$j]],\@prob_estimates);
    }
  }
  else{
    for(my $j=$begin;$j<$end;$j++){
      my $xSet = $self->{'problem'}{'x'}[$$perm[$j]];
      my @dec_values;
      $target[$$perm[$j]] = $self->predict($sub_model,$xSet,\@dec_values);
    }
  }
  return (\@target,$message);
}

sub group_classes { #(\$nr_class,\@label,\@start,\@count,\@perm);
  my ($self,$nr_class,$label,$start,$count,$perm) = @_;
  my @data_label;
  my $l = $self->{'problem'}{'count'};
  $$nr_class = 0;
  for(my $i = 0; $i < $l; $i++){
    my $this_label = $self->{'problem'}{'y'}[$i];
    my $j;
    for($j = 0; $j < $$nr_class; $j++){
      if($this_label eq $$label[$j]){
        $$count[$j]++; last;
      }
    }
    $data_label[$i] = $j;
    if($j eq $$nr_class){
      $$label[$$nr_class] = $this_label;
      $$count[$$nr_class] = 1;
      $$nr_class++;
    }
  }
  $$start[0] = 0;
  for(my $i=1;$i < $$nr_class; $i++){ $$start[$i] = $$start[$i - 1] + $$count[$i - 1]; }
  for(my $i=0;$i < $l; $i++){
    $$perm[$$start[$data_label[$i]]] = $i;
    $$start[$data_label[$i]]++;
  }
  $$start[0] = 0;
  for(my $i=1;$i < $$nr_class; $i++){ $$start[$i] = $$start[$i - 1] + $$count[$i - 1]; }
}

sub set_problem {
  my $self = $_[0];
  $self->{'problem'} = $_[1];
  $self->{'problem'}{'count'} = 1 * @{ $self->{'problem'}{'y'} };
}

sub read_problem_file {
  my $self = shift;
  my $model = shift;
  my @files = @_;
  my $max_index = 0;
  $self->{'problem'}{'count'} = 0;
  $self->{'problem'}{'elements'} = 0;

  my %keySet;
  foreach my $file (@files){ $self->_process_problem_file($file,\%keySet,\$max_index); }
  $self->{'keySetHash'} = \%keySet;
  
  my $gamma = 1 * $model->gamma();
  if($gamma == 0 and $max_index){
    $model->set_gamma(1/$max_index);
  }
}

sub _process_problem_file {
  my ($self,$file,$keySet,$max_index) = @_;
  if(!(-e $file)){ die "File $file does not exist.\n"; }
  my $result = join(' ',`file "$file"`);
  my $fh;
  if($result =~ /bzip2/){
    use IO::Uncompress::Bunzip2 qw(bunzip2 $Bunzip2Error) ;
    $fh = IO::Uncompress::Bunzip2->new($file) or die "bunzip2 failed: $Bunzip2Error\n";
  }
  elsif($result =~ /gzip/){
    use IO::Uncompress::Gunzip qw(gunzip $GunzipError) ;
    $fh = IO::Uncompress::Gunzip->new($file) or die "Gunzip failed: $GunzipError\n";
  }
  elsif($result =~ /ASCII/){
    open $fh, '<', $file or die "Unable to open $file.\n";
  }
  else{ die "Unhandled file type $result from $file"; }

  foreach my $line (<$fh>){
    my @items = split(/\s+/,$line);
    my $row = $self->{'problem'}{'count'};
    $self->{'problem'}{'y'}[$row] = shift @items;
    my %xSet;
    foreach my $item (@items){
      my ($x,$val) = split(/:/,$item);
      $xSet{$x} = $val;
      if($x > $$max_index){ $$max_index = $x; }
      $$keySet{$x} = 1;
    }
    $self->{'problem'}{'x'}[$row] = \%xSet;
    $self->{'problem'}{'elements'} += @items;
    $self->{'problem'}{'count'}++;
  }
  close $fh;
}

sub add_vector {
  my ($self,$y,$xSet) = @_;
  my $row = $self->{'problem'}{'count'};
  $self->{'problem'}{'y'}[$row] = $y;
  $self->{'problem'}{'x'}[$row] = $xSet;
  $self->{'problem'}{'count'}++;

}

sub save_problem_file {
  my ($self,$save_set_file) = @_;
  open my $input, '>', $save_set_file or die "Couldn't open save set file $save_set_file\n";
  for(my $row = 0;$row < $self->{'problem'}{'count'}; $row++){
    my @line = ($self->{'problem'}{'y'}[$row]);
    my $xSet = $self->{'problem'}{'x'}[$row];
    my @features = sort {$a <=> $b} keys %$xSet;
    foreach my $feature (@features){
      push @line, $feature.':'.$$xSet{$feature};
    }
    print $input join(' ',@line)."\n";
  }
  close $input;
}

sub problem_y_vectors { my ($self) = @_; return @{ $self->{'problem'}{'y'} }; }
sub num_test_vectors { my ($self) = @_; return $self->{'problem'}{'count'}; }

sub add_vector {
  my ($self,$y,$xSet) = @_;
  my $row = $self->{'problem'}{'count'};
  $self->{'problem'}{'y'}[$row] = $y;
  $self->{'problem'}{'x'}[$row] = $xSet;
  $self->{'problem'}{'count'}++;
  return $self->{'problem'}{'count'};
}

sub predictFile {
  my ($self,$out,$file,$model) = @_;
  my $correct = 0; my $count = 0;
  if(!(-e $file)){ die "File $file does not exist.\n"; }
  my $result = join(' ',`file "$file"`);
  my $fh;
  if($result =~ /bzip2/){
    use IO::Uncompress::Bunzip2 qw(bunzip2 $Bunzip2Error) ;
    $fh = IO::Uncompress::Bunzip2->new($file) or die "bunzip2 failed: $Bunzip2Error\n";
  }
  elsif($result =~ /gzip/){
    use IO::Uncompress::Gunzip qw(gunzip $GunzipError) ;
    $fh = IO::Uncompress::Gunzip->new($file) or die "Gunzip failed: $GunzipError\n";
  }
  elsif($result =~ /ASCII/){
    open $fh, '<', $file or die "Unable to open test_file $file.\n";
  }
  else{ die "Unhandled file type $result from $file"; }
  if($self->{'threads'}){
    my @threads;
    my @dump;
    while(my $line = <$fh>){ chomp($line);
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
        push @threads, threads->create('_predict_thr',$self,$model,\@dump);
        @dump = ();
        push @dump, $line;
      }
    }
    if(@dump){
      if(@threads > $self->{'maxthreads'}){
        my $thread = shift @threads;
        print $out $thread->join();
      }
      push @threads, threads->create('_predict_thr',$self,$model,\@dump);
    }
    foreach my $thread (@threads){
      print $out $thread->join();
    }
  }
  else{
    while(my $line = <$fh>){ chomp($line);
      my @items = split(/\s+/,$line);
      my $y = shift;
      my %xSet;
      foreach my $item (@items){
        if($item =~ /(\d+):(\S+)/){
          my $x = 1 * $1; my $val = 1 * $2;
          $xSet{$x} = $val;
        }
      }
      my @dec_values;
      my $y_predict = $self->predict($model,\%xSet,\@dec_values);
      print $out $y_predict."\n";
    }
  }
  close $fh;
}

sub _predict_thr {
  my ($self,$model,$lines) = @_;
  my $return;
  foreach my $line (@$lines){
    my @items = split(/\s+/,$line);
    my $y = shift;
    my %xSet;
    foreach my $item (@items){
      if($item =~ /(\d+):(\S+)/){
        my $x = 1 * $1; my $val = 1 * $2;
        $xSet{$x} = $val;
      }
    }
    my @dec_values;
    $return .= $self->predict($model,\%xSet,\@dec_values)."\n";
  }
  return $return;
}

sub predict {
  my ($self,$model,$xSet,$dec_values) = @_;
  my $gamma = $model->gamma();
  my $coef0 = $model->coef0();
  my $degree = $model->degree();
  my $keySet = $self->{'keySet'};
  if($model->svm_type() eq 'one_class' or
    $model->svm_type() eq 'epsilon_svr' or
    $model->svm_type() eq 'nu_SVR'){
    my $sum = 0;
    for(my $row = 0; $row < $model->count(); $row++){
      $sum += $model->sv_coef($row,0) * 
        _kernel($keySet,$xSet,$model->sv($row),$gamma,$coef0,$degree);
    }
    $sum -= $model->rho(0);
    if($model->svm_type() ne 'one_class'){ return $sum; }
    if($sum > 0){ return 1; }
    else{ return -1; }
  }
  else{
    if(!$dec_values){ my @dec_values_array; $dec_values = \@dec_values_array; }
    my @kvalues;
    for(my $row = 0; $row < $model->count(); $row++){
      $kvalues[$row] = _kernel($keySet,$xSet,$model->sv($row),$gamma,$coef0,$degree);
    }
    my @start;
    for(my $i = 1; $i < $model->nr_class(); $i++){
      $start[$i] = $start[$i - 1] + $model->nr_sv($i - 1);
    }
    my @vote;
    for(my $i = 0; $i < $model->nr_class(); $i++){ $vote[$i] = 0; }
    my $p = 0;
    for(my $i = 0; $i < $model->nr_class(); $i++){
      for(my $j = $i + 1; $j < $model->nr_class(); $j++){
        my $sum = 0;
        my $si = $start[$i];
        my $sj = $start[$j];
        my $ci = $model->nr_sv($i);
        my $cj = $model->nr_sv($j);
        my @coef1 = $model->sv_coef_class_vector($j - 1);
        my @coef2 = $model->sv_coef_class_vector($i);
        for(my $k = 0; $k < $ci; $k++){ $sum += $coef1[$si + $k] * $kvalues[$si + $k]; }
        for(my $k = 0; $k < $cj; $k++){ $sum += $coef2[$sj + $k] * $kvalues[$sj + $k]; }
        $sum -= $model->rho($p);
        $$dec_values[$p] = $sum;
        if($$dec_values[$p] > 0){ $vote[$i]++; }
        else{ $vote[$j]++; }
        $p++;
      }
    }
    my $vote_max_idx = 0;
    for(my $i = 1; $i < $model->nr_class(); $i++){
      if($vote[$i] > $vote[$vote_max_idx]){ $vote_max_idx = $i; }
    }
    return $model->label($vote_max_idx);
  }
}

sub _kernel_linear { return _dot(@_); } # (xSet,sv)
sub _kernel_polynomial {
  my ($keySet,$xSet,$sv,$gamma,$coef0,$degree) = @_;
  return ($gamma * _dot($keySet,$xSet,$sv) + $coef0) ** $degree;
}
sub _kernel_rbf {
  my ($keySet,$xSet,$sv,$gamma) = @_;
  my $sum = 0;
  foreach my $key (@{$keySet}){
    $sum += ( $$xSet{$key} - $$sv{$key} ) ** 2;
  }
  return exp(-1 * $gamma * $sum);
}
sub _kernel_sigmoid {
  my ($keySet,$xSet,$sv,$gamma,$coef0) = @_;
  return tanh($gamma * _dot($keySet,$xSet,$sv) + $coef0);
}

sub _dot{
  my ($keySet,$x,$y) = @_;
  my $sum = 0;
  foreach my $key (@{$keySet}){
    $sum += $$x{$key} * $$y{$key}; 
  }
  return $sum;
}

sub loger{
  my ($self,$message) = @_;
  if($self->{'debug'}){
    $message = "DBG: ".$message;
    if($self->{'debug_time'}){ $message = "[".time."] ".$message; }
    print STDERR $message."\n";
  }
}

sub initCoef {
  my ($self) = @_;
  if(!defined($self->{'cost'})){ $self->{'cost'} = 1; }
  if(!defined($self->{'nu'})){ $self->{'nu'} = 0.5; }
  if(!defined($self->{'epsilon_svr'})){ $self->{'epsilon_svr'} = 0.1; }
  if(!defined($self->{'epsilon_svr'})){ $self->{'epsilon_svr'} = 0.001; }
}

1;
