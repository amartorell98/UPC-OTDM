## Parameters
param m; # Data rows
param n; # Data columns
param nu; # Weighted objective parameter
param x{1..m, 1..n}; # Actual dataset points
param y{1..m}; # Actual dataset labels

## Variables
var w{1..n}; # Normal to the separation hyperplane, vars to optimize
var gamma; # Plane location with respect to the origin, intercept
var s{1..m}; # Slacks

## Objective function
minimize fobj: 0.5*sum{j in {1..n}}(w[j]^2) + nu*sum{i in {1..m}}s[i];

# Constraints
subject to soft_constraints {i in 1..m}:
	-y[i]*(sum{j in 1..n} (w[j]*x[i, j]) + gamma) - s[i] + 1 <= 0;
subject to slack_positiveness {i in 1..m}:
	-s[i] <= 0;