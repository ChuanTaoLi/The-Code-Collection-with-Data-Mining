# The-Code-Collection-with-Data-Mining
想把实践过的代码放进这个仓库里面，这里会把代码、结果和示例都收集起来。<br>

**LASSO回归**<br>
该文件夹里面存放LASSO回归特征筛选相关的程序文件以及示例数据、导出结果。运行LASSO.py可以计算LASSO特征筛选前后的方差膨胀因子、岭迹图的绘制、十折交叉验证选择最佳正则化系数以及导出数据集。<br>
LASSO回归的步骤如下：<br>
1. 首先 绘制岭迹图，通过观察各回归系数的收敛情况进行合适的正则化系数搜索区间的选择。
2. 接着，通过十折交叉验证进行最佳正则化系数的搜索。
3. 最后，计算特征筛选前后的方差膨胀因子。
4. 以及导出相应的文件。、
   
**贝叶斯优化超参数寻优**<br>
该文件夹里面存放TPE过程的贝叶斯优化，损失函数为最大化准确率，可以根据需要进行分类器或回归器的调整，其在训练集里面进行十折交叉验证进行最佳超参数组合的搜索，最后导出最佳超参数组合以及最佳损失函数值。<br>
LightGBM预测效果对比.py是调参前后对比用的。<br>

**过采样**<br>
该文件夹里面存放MAHAKIL的代码和示例。<br>
MAHAKIL可以应对二分类或多分类问题的过采样，但是要考虑多重共线性的影响，如何计算出来的行列式为0就运行不了了，主函数入口是MAHAKIL.py，validation.py是辅助文件。<br>
这里的代码对[MAHAKIL之最新类不平衡过采样方法](http://t.csdnimg.cn/IFdxs)的改进，由于该方法提出是2018年，0.22版本的sklearn有很多库现在没有了，所以对删除的库进行的替换。<br>
放进去的文献是MAHAKIL的开山之作。<br>

**后处理**<br>
该文件夹里面存放保序回归的代码和示例。<br>
根据标签数量调整即可，运行便能出结果，适合二分类和多分类。<br>
