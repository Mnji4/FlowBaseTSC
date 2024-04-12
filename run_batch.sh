echo "Begin"
python main_distilight.py --env_config ./config/config_196.json > log196.txt
echo "newyork finished."
python main_distilight.py --env_config ./config/config_hangzhou.json > log_hangzhou.txt
echo "hangzhou script finished."
python main_distilight.py --env_config ./config/config_jinan.json > log_jinan.txt
echo "jinan script finished."