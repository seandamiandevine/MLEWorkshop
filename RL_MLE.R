#' @author Sean Devine
#' @date 10-2021
#' 
#' For the intro to MLE tutorial presented at Concordia University, Montreal, 10-2021
#' First part of the code simulates data from a simple learning task using a delta-learning 
#' rule RL model with a softmax choice function.
#' Second part fits the model to the simulated data
#' Third and fourth part visualize the parameter recovery and nested model comparison respectively


# Simulate data -----------------------------------------------------------

# Simulation parameters
set.seed(2021)    # Random seed for replication
N = 10            # Number of subject
Nt  = 200         # Number of trials 
mu = c(1, -1)     # Mean payoff for choices 1 and 2 
sigma = 3         # SD of payoff distributions

# Visualize reward curves
curve(dnorm(x, mu[1], sigma), -10, 12, xlab='Reward', ylab = '', yaxt='n', col='blue')
curve(dnorm(x, mu[2], sigma), -10, 12, xlab='Reward', ylab = '', yaxt='n', col='purple', add=T)
legend('topright', bty='n', col=c('blue', 'purple'), pch = c(15, 17), legend=c('',''), pt.cex = 2)

lr = runif(N)              # True learning rates
invtemp = runif(N)         # True inverse temperatures

data = data.frame(stringsAsFactors = F) # initialize data frame
for(id in 1:N){
  
  cat('Simulating data for subject', id, 'out of', N, '\n')

  V = c(0, 0) # Initial expected value
  
  for(t in 1:Nt) {
    ev  = exp(invtemp[id]*V) # expected value
    sev = sum(ev)
    p   = ev/sev            # probability each choice
    
    # Simulate choice based on calculated probabilitu
    choice = sample(c(1,2), size = 1, prob = p)
    
    # Sample from normal distribution with mean = mu[choice]
    outcome <- round(rnorm(1, mean = mu[choice], sd = sigma), 2)
    
    # Delta-rule learning
    V[choice] <- V[choice] + lr[id] * (outcome - V[choice])
    
    data = rbind(data, data.frame(id=id, t=t, choice=choice, outcome=outcome, stringsAsFactors = F))
  }
 }

tapply(data$outcome, data$choice, mean)


# Fit ---------------------------------------------------------------------

# Prepare containers to store parameter estimates
a_hat = c()
b_hat = c()

# Loops through subjects
for(id in unique(data$id)) {
  
  cat('Fitting data for subject', id, 'out of', N, '\n')
  
  # Extract relevant variables for this subject 
  choices = data$choice[data$id==id]
  outcomes = data$outcome[data$id==id]
  trials = data$t[data$id==id]
  
  # Define the log-likelihood function used for MLE
  obfunc <- function(params)  {
    
    # Set up params with readable names
    alpha = params[1]
    beta = params[2]
    
    # Initialize expected value
    V <- c(0, 0)
    
    # loop through each trial and compute log-likelihood
    probs = c()
    for(t in trials) {
      # Softmax
      ev  = exp(beta*V)                # expected value
      sev = sum(ev)
      p   = ev/sev                     # probability each choice
      probs[t] = log(p[choices[t]])    # probability of the choice the subject actually made
      
      # Value update
      V[choices[t]] = V[choices[t]] + alpha*(outcomes[t] - V[choices[t]])
    }
    
    # return the summed (minus) log-likelihood, because optim minimizes by default
    return(-sum(probs))
  }
  
  
  # Use optim to minimize the (minus) log-likelihood function
  x0 = runif(2) # starting point for alpha and beta
  opt = optim(fn=obfunc, par=x0, method='L-BFGS-B', lower=c(0,0), upper=c(1, Inf), hessian=T)
  a_hat[id] = opt$par[1]
  b_hat[id] = opt$par[2]
  
  # Get 95% CI using Hessian matrix
  stderr = diag(sqrt(1/opt$hessian))
  upperbound = opt$par+1.96*stderr
  lowerbound = opt$par-1.96*stderr
  CI = matrix(c(lowerbound, upperbound), ncol=2, 
              dimnames = list(c('alpha', 'beta'),
                              c('2.5%', '97.5%')))
}


# Parameter Recovery ------------------------------------------------------

plot(a_hat, lr, xlab=expression(estimated~alpha), ylab=expression(true~alpha))
r = cor(a_hat,lr)
legend('topleft', bty='n', legend=paste0('r = ', round(r, 2)))

plot(b_hat, invtemp,xlab=expression(estimated~beta), ylab=expression(true~beta))
r = cor(b_hat, invtemp)
legend('topleft', bty='n', legend=paste0('r = ', round(r, 2)))


# Nested model comparison -------------------------------------------------
# You can run the above model with beta = 1 instead of as a free parameter to get LLreduced! 
# Just trust me for now that it is this

# Remember! These are negative LL, so lower is better (i.e., more likely)
LLreduced = 35.8733
LLfull = 35.71921
deviance = -2*(LLfull-LLreduced)
p = pchisq(deviance, 1, lower.tail = F)




