# PINNs for (2+1) D wave equation

$\alpha^2 \nabla^2 \phi+f=\frac{\partial^2 \phi}{\partial t^2}$

where $\nabla^2 \equiv \frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial z^2}$ is the Laplacian operator

## mainbody code
Code Improved from [NeuralPDE](https://github.com/SciML/NeuralPDE.jl)

### 1 forward problem
Setting $\alpha=1500$, setting $\phi$ as a network
```
n=10
chain_u = Lux.Chain(Dense(3,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,1))
```
### 2 inverse problem
Setting $\alpha$ and $\phi$ as two network
```
n = 10
chain_u = Lux.Chain(Dense(3,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,1))
chain_a = Lux.Chain(Dense(3,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,n,Lux.σ),Dense(n,1))    
```


