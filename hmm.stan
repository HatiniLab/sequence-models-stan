data {
    int<lower=1> K; // number of groups
    int<lower=1> N; // number of data points
    int<lower=1> T; // length of timeseries
    real y[N,T]; // observations
}
parameters {
    real<lower=0,upper=1> p0 ;     //initial prob grp 1
    real<lower=0,upper=1> TP[K] ;  //transition probs of staying in group
    real mu[K]; // locations of mixture components
    real<lower=0> sigma[K]; // scales of mixture components
}
transformed parameters {
    real<lower=0,upper=1> prob_grp[N,T];  //smoother estimate probability of group membership
    real<lower=0,upper=1> pred[N,T];   //one-step filter prediction of probabililty of group membership
    {
        real F[N,T];   //filter forwards group membership prob
        real B1[N,T];  //backwards information filter from grp 1
        real B2[N,T];  //backwards information filter from grp 2
        real Z1[N,T];  //intermediate data
        real Z2[N,T];  //intermediate data
        real like1;    
        real like2;
        real p1;
        real p2;
        real k;
        int i;

        //Forwards algorithm
        for (n in 1:N) { 
            F[n,1]<-p0; 
            pred[n,1]<-F[n,1];
        }
        for (t in 1:T){
            for (n in 1:N) {
                //update prior using data
                like1<-exp(normal_log(y[n,t],mu[1],sigma[1]));
                like2<-exp(normal_log(y[n,t],mu[2],sigma[2]));
                p1<-F[n,t]*like1;
                p2<-(1-F[n,t])*like2;
                F[n,t]<-p1/(p1+p2);
    
                //predict forward one timestep
                if (t != T) {
                    p1<-F[n,t]*TP[1]+(1-F[n,t])*(1-TP[2]);
                    p2<-F[n,t]*(1-TP[1])+(1-F[n,t])*TP[2];
                    F[n,t+1]<-p1/(p1+p2);
                    pred[n,t+1]<-F[n,t+1];
                }
            }
        }  
        //backwards algorithm
        for (n in 1:N) { 
            B1[n,T]<-1; 
            B2[n,T]<-1; 
        }
        for (t in 1:(T-1)){
            i<-t*(-1)+10;      // transform t to get a backwards loop
            for (n in 1:N){
                like1<-exp(normal_log(y[n,i+1],mu[1],sigma[1]));
                like2<-exp(normal_log(y[n,i+1],mu[2],sigma[2]));  
        
                B1[n,i]<-TP[1]*like1*B1[n,(i+1)]+(1-TP[2])*like2*B2[n,(i+1)];
                B2[n,i]<-(1-TP[1])*like1*B1[n,(i+1)]+TP[2]*like2*B2[n,(i+1)];
        
                k<-B1[n,i]+B2[n,i];
                B1[n,i]<-B1[n,i]/k;
                B2[n,i]<-B2[n,i]/k;
            }
        }
        // put it all together
        for (t in 1:T){
            for (n in 1:N) {
                Z1[n,t]<-F[n,t]*B1[n,t];
                Z2[n,t]<-(1-F[n,t])*B2[n,t];
                prob_grp[n,t]<-Z1[n,t]/(Z1[n,t]+Z2[n,t]);
            }
        }
    }
}
model {
    real ps; // temp for log component densities
    sigma ~ cauchy(0,2.5);
    mu ~ normal(0,10);

    for (t in 1:T){
        for (n in 1:N) {
            ps <- pred[n,t]*exp(normal_log(y[n,t],mu[1],sigma[1])) + (1-pred[n,t])*exp(normal_log(y[n,t],mu[2],sigma[2]));
              increment_log_prob(log(ps));
        }
    }
}