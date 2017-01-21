#Kaggle Coupon Purchase Prediction Solution
This is the solution to the kaggle challenge [Ponpare Coupon Purchase Prediction](https://www.kaggle.com/c/coupon-purchase-prediction). In this competition, we use past purchase and browsing behavior to predict which coupons a customer will buy in a given future period of time.

We are provided with a year of transactional data for 22,873 users on the site ponpare.jp. The training set spans the dates 2011-07-01 to 2012-06-23. The test set spans the week after the end of the training set, 2012-06-24 to 2012-06-30. The goal of the competition is to recommend a ranked list of coupons for each user in the dataset (found in user_list.csv). Our predictions are scored against the actual coupon purchases, made during the test set week, of the 310 possible test set coupons.

Submissions are evaluated according to the Mean Average Precision @ 10 (MAP@10):
<p align="center"><img src="http://latex.codecogs.com/gif.latex?MAP@10%20=%20\frac{1}{|U|}%20\sum_{u=1}^{|U|}%20\frac{1}{min(m,%2010)}%20\sum_{k=1}^{min(n,10)}%20P(k)" border="0" align="center"/></p>
where |U| is the number of users, P(k) is the precision at cutoff k, n is the number of predicted coupons, and m is the number of purchased coupons for the given user. If m = 0, the precision is defined to be 0.

To solve this challenge, we will first explore the raw data and implement a data processing pipeline to process data and engineer the features. This step is proved to be most important in most kaggle competitions. Since everyone has the accessibility to advanced algorithm such as Xgboost, the real difference between kagglers are data processing and feature engineering. After proper data processing and feature engineering we will use a three layer multi-layer perceptron (MLP) to regress each coupon's purchase probability.

##Input Data Pipeline
To synthesize the various raw data, we will consider a (user,coupon) pair with label equal to 1 if the user purchased the coupon and label equal to 0 if didn't. This label can be interpreted as purchasing probability as well. For each (user,coupon) pair, its feature consists of user profile (sex and age), coupon profile('CATEGORY_NAME','PRICE_RATE',...etc.) and user purchase history. Here, user purchase history is defined as the average of coupon profiles the particular user purchased before this coupon. This can be treated as user behaviour profile which is simply the statistics of his/her purchase history. Therefore, if two coupons went on sale at different times, the purchase history feature could be totally different for a user.

All the features mentioned above are somewhat conventional processing of raw data. However, in this competition, there is a weird thing that many coupons have been browsed by some users even before its official valid time! Ponpare didn't explain anything about this problem but as discussed in kaggle forum, many kagglers believed that those coupons' website may go online well before its sale starting time. Therefore some users may have chance to browse it (called preview in my code) before it should be displayed. We find that many users will buy this coupon later on if they preview it. We guess this is because Ponpare has its own recommendation system and will recommend user some coupon based on his/her purchase history, even when this coupon's website just goes online and it is not valid for sale yet. Those users will be notified with their favorite coupons in advance and therefore it is highly probable that they will buy it later on.

We did not know if this was the feature that was valid to use. We guessed that in a real online recommendatition system this feature should not be used. However, for the sake of this comptetion, we decided to use it unless Ponpare officially admit that is some kind of invalid expolration of data.

Another problem is that since most users just buy little coupons, many (user,coupon) pairs are labeled with 0. We could not use all of them as negative examples. Therefore, we will define a neg/pos ratio and sampling negative example based on this ratio. Therefore, ensemble model with different samples of negative examples will greatly improve the final score on leader board.

##MLP Training
In this solution, I use mxnet to build a three layer MLP with dropout to regress the purchase probability. The net visualization is:
<p align="center"><img src="https://github.com/ssw5075839/coupon/blob/master/pics/Three%20Layer%20MLP.PNG" border="0" align="center"/></p>

