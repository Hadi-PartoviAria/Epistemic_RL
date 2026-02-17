outpath=$(pwd)

if ! command -v srun &> /dev/null
then
  echo "SLURM not found. Running experiments sequentially."
  try_parallel() {
    echo $@
    $@
  }
else
  echo "Running experiments with SLURM."
  try_parallel() {
    echo $@
    srun $@ > /dev/null &
    sleep 1
  }
fi

find_and_cd_to_impl_dir() {
  cd impl &> /dev/null ||
  cd ../impl &> /dev/null ||
  cd ../../impl &> /dev/null ||
  cd ../../../impl &> /dev/null ||
  cd ../../../../impl &> /dev/null
}

single_experiment() {(
  # $1: input filename
  # $2: output filename
  # ${@:3}: parameters
  find_and_cd_to_impl_dir
  try_parallel ./rlbb $1 -o $2 ${@:3}
)}

emm_experiment() {
  # $1: input filename
  # $2: output filename
  # ${@:3}: parameters
  for fn in expected mean min; do
    outname=$2_$fn
    single_experiment $1 $outname -f $fn ${@:3}
  done
}

frompath() {  
  # $1: path (relative to base path)
  # $2: experiment command (with $1: in $2: out, ${@:3}:params)
  # $3: filename without json ending
  # $4: suffix for output
  # ${@:5}: arguments
  input=../$1/$3.json
  output=$outpath/$3$4
  $2 $input $output ${@:5}
}
