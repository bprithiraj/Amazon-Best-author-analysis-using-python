# Amazon-Best-author-analysis-using-python
A simple Data Analysis Project using Python fundamentals,Numpy,Pandas and Matpotlib.No ML algos have been used in this project.

In linear regression we tried to predict the value of y(i)
 for the i
‘th example x(i)
 using a linear function y=hθ(x)=θ⊤x.
. This is clearly not a great solution for predicting binary-valued labels (y(i)∈{0,1})
. In logistic regression we use a different hypothesis class to try to predict the probability that a given example belongs to the “1” class versus the probability that it belongs to the “0” class. Specifically, we will try to learn a function of the form:

P(y=1|x)P(y=0|x)=hθ(x)=11+exp(−θ⊤x)≡σ(θ⊤x),=1−P(y=1|x)=1−hθ(x).
 
The function σ(z)≡11+exp(−z)
 is often called the “sigmoid” or “logistic” function – it is an S-shaped function that “squashes” the value of θ⊤x
 into the range [0,1]
 so that we may interpret hθ(x)
 as a probability. Our goal is to search for a value of θ
 so that the probability P(y=1|x)=hθ(x)
 is large when x
 belongs to the “1” class and small when x
 belongs to the “0” class (so that P(y=0|x)
 is large). For a set of training examples with binary labels {(x(i),y(i)):i=1,…,m}
 the following cost function measures how well a given hθ
 does this:

J(θ)=−∑i(y(i)log(hθ(x(i)))+(1−y(i))log(1−hθ(x(i)))).
Note that only one of the two terms in the summation is non-zero for each training example (depending on whether the label y(i)
 is 0 or 1). When y(i)=1
 minimizing the cost function means we need to make hθ(x(i))
 large, and when y(i)=0
 we want to make 1−hθ
 large as explained above. For a full explanation of logistic regression and how this cost function is derived, see the CS229 Notes on supervised learning.

We now have a cost function that measures how well a given hypothesis hθ
 fits our training data. We can learn to classify our training data by minimizing J(θ)
 to find the best choice of θ
. Once we have done so, we can classify a new test point as “1” or “0” by checking which of these two class labels is most probable: if P(y=1|x)>P(y=0|x)
 then we label the example as a “1”, and “0” otherwise. This is the same as checking whether hθ(x)>0.5
.

To minimize J(θ)
 we can use the same tools as for linear regression. We need to provide a function that computes J(θ)
 and ∇θJ(θ)
 for any requested choice of θ
. The derivative of J(θ)
 as given above with respect to θj
 is:

∂J(θ)∂θj=∑ix(i)j(hθ(x(i))−y(i)).
Written in its vector form, the entire gradient can be expressed as:

∇θJ(θ)=∑ix(i)(hθ(x(i))−y(i))
This is essentially the same as the gradient for linear regression except that now hθ(x)=σ(θ⊤x)
.
