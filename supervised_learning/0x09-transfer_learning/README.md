# Learning Objectives

At the end of this project, you are expected to be able to explain to anyone, without the help of Google:

## General

What is a transfer learning?

![traditional ML vs Transfer Learning](https://miro.medium.com/max/1400/1*9GTEzcO8KxxrfutmtsPs3Q.png)

> Situation where what has been learned in one setting is exploited to improve generalization in another setting.

- leverage knowledge from pre-trained models and use it to solve new problems
-  In transfer learning, you can leverage knowledge (features, weights etc) from previously trained models for training newer models and even tackle problems like having less data for the newer task!
- knowledge from an existing task acts as an additional input when learning a new target task.

> During the process of transfer learning, the following three important questions must be answered:
> What to transfer: This is the first and the most important step in the whole process. We try to seek answers about which part of the knowledge can be transferred from the source to the target in order to improve the performance of the target task. When trying to answer this question, we try to identify which portion of knowledge is source-specific and what is common between the source and the target.
> When to transfer: There can be scenarios where transferring knowledge for the sake of it may make matters worse than improving anything (also known as negative transfer). We should aim at utilizing transfer learning to improve target task performance/results and not degrade them. We need to be careful about when to transfer and when not to.
> How to transfer: Once the what and when have been answered, we can proceed towards identifying ways of actually transferring the knowledge across domains/tasks. This involves changes to existing algorithms and different techniques, which we will cover in later sections of this article. Also, specific case studies are lined up in the end for a better understanding of how to transfer.

![transfer learning](https://ruder.io/content/images/2017/03/transfer_learning_setup.png)

![different domains](https://miro.medium.com/max/1400/1*vE8VO6isG0fSVYzgci3DuQ.png)

### Transfer Learning Strategies

![strategies](https://miro.medium.com/max/1222/1*mEHO0-LifV7MgwXSpY9wyQ.png)

- categorized based on the type of traditional ML algorithms involved

> Inductive Transfer learning: In this scenario, the source and target domains are the same, yet the source and target tasks are different from each other. The algorithms try to utilize the inductive biases of the source domain to help improve the target task. Depending upon whether the source domain contains labeled data or not, this can be further divided into two subcategories, similar to multitask learning and self-taught learning, respectively.
> Unsupervised Transfer Learning: This setting is similar to inductive transfer itself, with a focus on unsupervised tasks in the target domain. The source and target domains are similar, but the tasks are different. In this scenario, labeled data is unavailable in either of the domains.
> Transductive Transfer Learning: In this scenario, there are similarities between the source and target tasks, but the corresponding domains are different. In this setting, the source domain has a lot of labeled data, while the target domain has none. This can be further classified into subcategories, referring to settings where either the feature spaces are different or the marginal probabilities.

![summary](https://miro.medium.com/max/1400/1*ZEJeJS06czdyPwov5EbCuQ.png)

### what to transfer across these categories

some of the following approaches can be applied:

> Instance transfer: Reusing knowledge from the source domain to the target task is usually an ideal scenario. In most cases, the source domain data cannot be reused directly. Rather, there are certain instances from the source domain that can be reused along with target data to improve results. In case of inductive transfer, modifications such as AdaBoost by Dai and their co-authors help utilize training instances from the source domain for improvements in the target task.
> Feature-representation transfer: This approach aims to minimize domain divergence and reduce error rates by identifying good feature representations that can be utilized from the source to target domains. Depending upon the availability of labeled data, supervised or unsupervised methods may be applied for feature-representation-based transfers.
> Parameter transfer: This approach works on the assumption that the models for related tasks share some parameters or prior distribution of hyperparameters. Unlike multitask learning, where both the source and target tasks are learned simultaneously, for transfer learning, we may apply additional weightage to the loss of the target domain to improve overall performance.
> Relational-knowledge transfer: Unlike the preceding three approaches, the relational-knowledge transfer attempts to handle non-IID data, such as data that is not independent and identically distributed. In other words, data, where each data point has a relationship with other data points; for instance, social network data utilizes relational-knowledge-transfer techniques

![Transfer Learning for Deep Learning](https://miro.medium.com/max/1400/1*Ww3AMxZeoiB84GVSRBr4Bw.png)

![inductive learning vs inductive transfer](https://miro.medium.com/max/1210/1*yjBaWnApTg_4Mz1xrHBJZg.png)

What is fine-tuning?

- selectively retrain some of the previous layers.

![hierchy of DNN](https://miro.medium.com/max/1400/1*BBZGHtI_vhDBeqsIbgMj1w.png)

- the initial layers have been seen to capture generic features, while the later ones focus more on the specific task at hand.
- we utilize the knowledge in terms of the overall architecture of the network and use its states as the starting point for our retraining step.
- achieve better performance with less training time

![Fine Tuning](https://miro.medium.com/max/1400/1*Lut6GJXQ6dGhnH9gQJQxgg.png)

![Fine Tuning vs Freezing](https://miro.medium.com/max/1400/1*AUI4rH8_tbb7x4xkBsHu2Q.png)

What is a frozen layer? How and why do you freeze a layer?

How to use transfer learning with Keras applications
