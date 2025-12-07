chmod +x lr_sweep.sh
./lr_sweep.sh --epochs 50 --seed 42

# 2. 生成对比图表 (横坐标: 学习率, 纵坐标: 测试准确率)
python plot_lr_sweep.py --seed 42 --output lr_sweep.png

# 3. 查看结果表格
python plot_lr_sweep.py --seed 42 --no-plot