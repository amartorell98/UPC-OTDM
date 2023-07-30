param m;
param k;

param d{1..m, 1..m};
var x{1..m, 1..m} binary;

minimize f: sum{i in 1..m, j in 1..m} d[i,j]*x[i,j];
subject to every_point_belongs_to_one_cluster{i in 1..m}: sum{j in 1..m} x[i,j] = 1;
subject to exactly_k_cluster: sum{j in 1..m} x[j,j] = k;
subject to a_point_can_only_belong_if_cluster_exists{i in 1..m, j in 1..m}: x[j,j] >= x[i,j];

