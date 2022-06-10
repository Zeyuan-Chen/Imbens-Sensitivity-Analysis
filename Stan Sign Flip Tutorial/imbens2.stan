data {
    int N;   
    int likelihood;
    real Y[N];
    vector[N] X;
    int W[N];
 
}


               
parameters {
    real beta_xy;
    real beta_xw;
    real alpha;
    real delta;
    real tau;
    real<lower=0> sigma;
   
}




model {

    //priors
    beta_xw ~ normal(0,3);
    beta_xy ~ normal(0,3);
    alpha ~ normal(0,3);
    delta ~ normal(0,3);

    for(i in 1:N){
        target += log_sum_exp(  bernoulli_logit_lpmf(W[i] | beta_xw*X[i]) + normal_lpdf(Y[i]|beta_xy*X[i]+tau*W[i],sigma),
                                bernoulli_logit_lpmf(W[i] | beta_xw*X[i] + alpha) + normal_lpdf(Y[i]|beta_xy*X[i]+tau*W[i]+ delta,sigma)
                          );
    }
    
        
     

}


