# 实验二 特征选择与特征提取

代码说明：

-   `task2_feature/scores.py`: 特征评价准则的实现，包括基于类内类间距离的可分性判据(`euclidean_distance`)、基于概率分布的散度距离的可分性判据(`divergence`)、基于t-检验的可分性判据(`t_test`)
-   `task2_feature/algorithms.py`: 特征选择的算法的实现，包括分枝定界法(`branch_and_bound`)、单独最优特征(`single_best`)、SFS(`sfs`)、SBS(`sbs`)
-   `task2_feature/test.py`: 在所给数据集上进行各种特征选择与提取方法实验，可复现报告中所有的数据和图表

运行步骤：

-   `python read_data.py`
-   `cd task2_feature`
-   `python test.py`

