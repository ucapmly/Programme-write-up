
using NeuralPDE, Lux, ModelingToolkit, Optimization, OptimizationOptimJL, OrdinaryDiffEq, Plots, OptimizationOptimisers,CUDA, Random, Plots, JLD2
import ModelingToolkit: Interval, infimum, supremum

# load data

## time snapshot data
n=128
t1=fill(1,n^2)
t2=fill(10,n^2)
t3=fill(200,n^2)
x_=vcat([j for j in 1:n for i in 1:n])
z_=vcat([j for i in 1:n for j in 1:n]) 
var_snapshot_t1=permutedims(hcat(t1,hcat(x_,z_)))
var_snapshot_t2=permutedims(hcat(t2,hcat(x_,z_)))
var_snapshot_t3=permutedims(hcat(t3,hcat(x_,z_)))
                                


### 3 source snapshot data
data3_1=readlines("datai3_1.txt")
data3_2=readlines("datai3_2.txt")
data3_3=readlines("datai3_3.txt")
u_3_1=[parse(Float64,data3_1[i]) for i in range(1,length = 16384)]      
u_3_2=[parse(Float64,data3_2[i]) for i in range(1,length = 16384)]   
u_3_3=[parse(Float64,data3_3[i]) for i in range(1,length = 16384)];
                
###  snapshot plots    

xs=1:1:128
zs=1:1:128


p11=plot(xs, zs, u_3_1, linetype=:contourf)
savefig("ture_3source_1.png")  
p22=plot(xs, zs, u_3_2, linetype=:contourf)
savefig("ture_3source_2.png")  
p33=plot(xs, zs, u_3_3, linetype=:contourf)
savefig("ture_3source_3.png") 

plot(p11,p22,p33)
savefig("p_all.png")  
       
## sensor data
sen_data=readlines("sen_datai_3.txt")
u_sen=[parse(Float64,sen_data[i]) for i in range(1,length = length(sen_data))] 

x_sen=vcat([j for i in 1:2 for j in 10:40],[10 for i in 0:(120-80)], [40 for i in 0:(120-80)])
z_sen=vcat([80 for i in 10:40], [120 for i in 10:40],[j for i in 1:2 for j in 80:120])
t_sen=[j for j in 1:256 for i in 1:length(z_sen)]
x_sen=repeat(x_sen,256)
z_sen=repeat(z_sen,256)                                            
var_sen=permutedims(hcat(t_sen,hcat(x_sen,z_sen)));

## free surface data
free_data=readlines("free_datai_3.txt")
u_free=[parse(Float64,free_data[i]) for i in range(1,length = length(free_data))] 

x_free=repeat([i for i in range(1,length = n)],256)
z_free=fill(1,n*256)
t_free=[j for j in 1:256 for i in 1:length(z_free)/256]
var_free=permutedims(hcat(t_free,hcat(x_free,z_free)));
 




# pde_system
                                                                                    
@parameters t x z 
@variables u(..), a(..)
Dt  = Differential(t)
Dtt = Differential(t)^2
Dxx = Differential(x)^2
Dzz = Differential(z)^2


t_min= 1.
t_max = 256.
x_min = 1.
x_max = 128.
z_min = 1.
z_max = 128.

## 2D PDE with a=1500
eq  = Dtt(u(t,x,z)) ~ (a(t,x,z))^2 * (Dxx(u(t,x,z)) + Dzz(u(t,x,z)))


## Initial and boundary conditions
bcs = [u(t,x,z_max) ~ 0]
#u(t_min,x,z) ~ 0.5*exp(-100*((x - 0.5)^2 + (y - 0.5)^2)) + 0.3*exp(-100*((x - 1.5)^2 + (y - 1.5)^2)),


## Space and time domains
domains = [t ??? Interval(t_min,t_max),
           x ??? Interval(x_min,x_max),
           z ??? Interval(z_min,z_max)]


@named pde_system = PDESystem(eq,bcs,domains,[t,x,z],[u(t,x,z),a(t,x,z)])



# Neural Network

## sampling interval of PDE internal point  
dt = 1.

## 4 hidden layer
input_ = length(domains)
n = 50 
chain_u = Lux.Chain(Dense(input_,n,Lux.??),Dense(n,n,Lux.??),Dense(n,n,Lux.??),Dense(n,n,Lux.??),Dense(n,1))
chain_a = Lux.Chain(Dense(input_,n,Lux.??),Dense(n,n,Lux.??),Dense(n,n,Lux.??),Dense(n,1))    

# additional_loss: sensor, free surface, snapshot.                                                                                     

function additional_loss(phi, ?? , p)
    return 10*sum(abs2, phi[1](var_snapshot_t1, ??.depvar.u).-u_3_1)/length(u_3_1)+10*sum(abs2, phi[1](var_snapshot_t2, ??.depvar.u).-u_3_2)/length(u_3_2)+10*sum(abs2, phi[1](var_snapshot_t3, ??.depvar.u).-u_3_3)/length(u_3_3)+sum(abs2, phi[1](var_sen, ??.depvar.u).-u_sen)/length(u_sen)+sum(abs2, phi[1](var_free, ??.depvar.u).-u_free)/length(u_free)
end

# PhysicsInformedNN
discretization = NeuralPDE.PhysicsInformedNN([chain_u,chain_a],NeuralPDE.GridTraining(dt), additional_loss=additional_loss)

#Optimization problem
prob = NeuralPDE.discretize(pde_system,discretization)

# Train Optimization problem
## train via BFGS
callback = function (p,l)
    println("Current loss is: $l")
    return false
end
                                                                                        
res = Optimization.solve(prob, BFGS();callback = callback,maxiters=5000)

# Train again via Adam
prob = remake(prob,u0=res.u)
res = Optimization.solve(prob,Adam(0.001);callback = callback,maxiters=5000)
                                                                                        
# result and plots

## weight and bias                                     
phi_u=discretization.phi[1];
phi_a=discretization.phi[2]

## u
u_pr_1=phi_u(var_snapshot_t1,res.u.depvar.u)
u_pr_11=[u_pr_1[i] for i in range(1,length = length(u_pr_1))]

u_pr_2=phi_u(var_snapshot_t2,res.u.depvar.u)
u_pr_22=[u_pr_2[i] for i in range(1,length = length(u_pr_2))]

u_pr_3=phi_u(var_snapshot_t3,res.u.depvar.u)
u_pr_33=[u_pr_3[i] for i in range(1,length = length(u_pr_3))]

## plots

xs=1:1:128
zs=1:1:128

### prediction
p1=plot(xs, zs, u_pr_11, linetype=:contourf,title = "t=1") 
savefig("pre1.png")       
p2=plot(xs, zs, u_pr_22, linetype=:contourf,title = "t=10") 
savefig("pre2.png")       
p3=plot(xs, zs, u_pr_33, linetype=:contourf,title = "t=200") 
savefig("pre3.png") 

### Misfit
p1=plot(xs, zs, u_3_1-u_pr_11, linetype=:contourf,title = "t=1") 
savefig("Mis1.png")       
p2=plot(xs, zs, u_3_2-u_pr_22, linetype=:contourf,title = "t=10") 
savefig("Mis2.png")       
p3=plot(xs, zs, u_3_3-u_pr_33, linetype=:contourf,title = "t=200") 
savefig("Mis3.png") 