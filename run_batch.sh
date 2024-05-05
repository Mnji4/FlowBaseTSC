echo "Begin"
for gamma in {0..5}
do
    echo "python main_traj.py --traj_gamma 0.$gamma > logs/jinan_gamma_0.$gamma.log"
    python main_traj.py --traj_gamma 0.$gamma >> logs/jinan_gamma_0.$gamma.log
done
echo "jinan script finished."