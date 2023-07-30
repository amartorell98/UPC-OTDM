## Parameters
param m; # Data rows
param n; # Data columns
param nu; # Weighted objective parameter
param x{1..m, 1..n}; # Actual dataset points
param y{1..m}; # Actual dataset labels

## Variables
var w{1..n}; # Normal to the separation hyperplane, vars to optimize
var lambda{1..m};

## Objective function
maximize fobj: sum{i in {1..m}}(lambda[i]) - 0.5*sum{i in {1..m}, j in {1..m}}(lambda[i]*y[i]*lambda[j]*y[j]*(sum{k in 1..n}(x[i,k]*x[j,k])));

# Constraints
subject to lambda_constraints: 
	sum{i in 1..m}(lambda[i]*y[i]) = 0;
subject to var_constraints {i in 1..m}:
	0 <= lambda[i] <= nu;

# Retrieve w
subject to get_w_values {i in 1..n}:
	w[i] = sum{j in 1..m}(lambda[j]*y[j]*x[j,i]);