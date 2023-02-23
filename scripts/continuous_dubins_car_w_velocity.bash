#!/bin/bash

failcount=0
loopcount=1

# options:
# -c - number of times to loop
# -p - environment; either continuous_dubins_car_w_velocity or continuous_dubins_car



while getopts c:p: option
  do
    case "$option"
    in
      c) loopcount=$OPTARG;;
      p) env=$OPTARG;;
    esac
  done

echo $loopcount
echo $env

#declare -a StringArray=("no-ob-1" "no-ob-2" "no-ob-3" "no-ob-4" "no-ob-5"  "ob-1" "ob-2" "ob-3" "ob-4" "ob-5" "ob-6"  "ob-7" "ob-8" "ob-9" "ob-10" "ob-11" "u" "cave" "cave-mini")


for i in {1..$loopcount};
do
  for map in "no-ob-1" "no-ob-2" "no-ob-3" "no-ob-4" "no-ob-5"  "ob-1" "ob-2" "ob-3" "ob-4" "ob-5" "ob-6"  "ob-7" "ob-8" "ob-9" "ob-10" "ob-11" "u" "cave" "cave-mini";
  do  
    rosrun turtlebotwrapper planwrapper.py --planner cont-sogbofa --env=$env --noise=True --map_name $map
    result=$?
    echo "RESULT =" $result
    if [ "$result" != "0" ]
    then
      $failcount+=1
      echo "TEST with map $map FAILED"
    fi
    echo "Succesfully Solved map =" $map
  done
done

echo $failcount " FAILURES / " $loopcount



