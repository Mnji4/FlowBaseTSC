echo "Begin"
# for gamma in {0..9}
# do
#     command="python main_traj.py --gamma 0.$gamma > logs/jinan_1.7_gamma_0.$gamma.log"
#     eval $command
# done

for gamma in {0..9}
do
    command="python main_traj.py --traj_gamma 0.$gamma > logs/jinan_1.7_trajgamma_0.$gamma.log"
    eval $command
done
echo "jinan script finished."