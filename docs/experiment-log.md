# LOG



# Question
1. ANN 是什么？
2. bi-encoder是什么？非对称的bi-encoder是什么？分别怎么得到的？分别怎么训练的？随便什么loss都可以训练嘛？是可以一开始用shared medical text encoder 然后两边分别加一个ontology-specific adapter 或 ontology prompt token就行吗？
3. 非对称bi-encoder的 非对称打分是什么？非对称 projection是什么？
4. 为什么codex每次他都要reconnecting5次
5. Codex对于SSD的伤害比较高，怎么解决？
6. topology-mismatch miner、directional calibration 或 reranker feedback。是什么意思？【像我当前的这个模型，其实非常普通，没有什么创新点，主要是构造正负样本之间的形态，创新都是在于数据的结构的变化上，而非在模型的结构的变化上】
7. 

```text
1. 我不太确定是一个功能一个功能的实现，还是说是整体的交给codex去运行。我觉得一个功能一个功能的实现可能会好一些
2. text_views.py我有点忘记了，之前构建这3个视图是怎么放到模型中的？
3. ontology_id 到底指的是每个term的ID，还是指的是ICD 10是一个IDICD，11是一个ID呢
4. 多正例训练 label 规则还需要定。现在确实存在一对一和一对多的关系,到底该怎么训练我还没有想好，哪种是最简单的呢.
5. 现在只想跑一个loss.
```

