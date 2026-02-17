for m in 1 2; do
  for n in $(seq 3 12); do
    python wumpus.py $n $n $m > wumpus_${n}_${n}_${m}.json
  done
done
