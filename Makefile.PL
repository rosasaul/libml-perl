use ExtUtils::MakeMaker;

WriteMakefile(
  'NAME'          => 'ML::SVM',
  'VERSION_FROM'  => 'lib/ML/SVM.pm',
  'PREREQ_PM'     => {
    # Require Threading
    'threads' => 0,
    'threads::shared' => 0,
    'Math::Trig' => 0,
    'Getopt::Long' => 0,
    'Storable' => 0,
    'IO::Compress::Bzip2' => 0
  },
  'ABSTRACT_FROM' => 'lib/ML/SVM.pm',
  'AUTHOR'        => 'Saul Rosa <rosa.saul@megaframe.org>'
);
