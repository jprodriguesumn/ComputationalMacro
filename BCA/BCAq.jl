#using ForwardDiff
using Calculus
using Roots
using Optim
using Plots



#################################################################
#
# Steady state conditions
#
#################################################################
"""
inputs:
x0: vector = (k0,l0) - steady state capital,labor guess
exogss: exogenous states = log(z),τl,τx,log(g)
params: parameters of the model = gn,gz,β,δ,ψ,σ,θ
tol: tolerance for newton 
maxn: number of newton iterations

output:
kss: ss capital
lss: ss labor
css: ss consumption
xss: ss investment
f: ss production

"""

function equilibrium_ss(x0,exogss,params,tol = 1e-12,maxn = 20)
    gn,gz,β,δ,ψ,σ,θ = params
    z,τl,τx,g = exp(exogss[1]),exogss[2],exogss[3],exp(exogss[4])
    βhat = β*(1.0+gz)^(-σ)
    kss,lss,css,f,xss = 0.0,0.0,0.0,0.0,0.0
    
    step = zeros(2)
    ss = zeros(2)
    res = zeros(2) 
    dres = zeros(2,2)
        
    ## Begin iteration
    for i = 1:maxn
        kss,lss = x0[1],x0[2]
        if kss < 0.0 
            println("Capital negative")
        end
        if lss < 0.0 
            println("labor negative")
        end
        f = kss^θ*(z*lss)^(1.0-θ)
        css = f - (1.0 + gz)*(1.0 + gn)*kss + (1.0 - δ)*kss - g
        if css<0.0
            println("consumption negative")
        end
        xss = f - css - g 

        ∂f∂k = θ * kss^(θ-1.0) * (z*lss)^(1.0-θ)
        ∂f∂l = (1.0-θ) * kss^θ*(z*lss)^(-θ) * z
        ∂f∂l∂l = -θ*(1.0-θ) * kss^θ*(z*lss)^(-θ-1.0) * z^2.0
        ∂f∂l∂k = θ*(1.0-θ) * kss^(θ-1.0)*(z*lss)^(-θ) * z
        ∂f∂k∂l = (1.0-θ) * θ * kss^(θ-1.0) * (z*lss)^(-θ) * z
        ∂f∂k∂k = (θ-1.0) * θ * kss^(θ-2.0) * (z*lss)^(1.0-θ)

        ∂css∂l = ∂f∂l
        ∂css∂k = ∂f∂k - (1.0 + gz)*(1.0 + gn) + (1.0 - δ)

        mris = (ψ*css) - (1-lss)*(1-τl)*∂f∂l
        ee = css*(1.+τx) - css*βhat*(∂f∂k + (1.0-δ)*(1.0 + τx))

        ∂mris∂l = ψ*∂css∂l + (1-τl)*∂f∂l - (1-lss)*(1-τl)*∂f∂l∂l
        ∂mris∂k = ψ*∂css∂k - (1-lss)*(1-τl)*∂f∂l∂k
        ∂ee∂l = (1.+τx) * ∂css∂l - ∂css∂l*βhat*(∂f∂k + (1.0-δ)*(1.0 + τx)) - css*βhat*∂f∂k∂l
        ∂ee∂k = (1.+τx) * ∂css∂k - ∂css∂k*βhat*(∂f∂k + (1.0-δ)*(1.0 + τx)) - css*βhat*∂f∂k∂k

        res[1],res[2] = mris,ee
        dres[1,1],dres[1,2] = ∂mris∂k, ∂mris∂l
        dres[2,1],dres[2,2] = ∂ee∂k, ∂ee∂l
        
        ## updating newton step
        step = -dres \ res
        x0 = x0 + step
        if norm(step) < tol
            break
        end
    end       
    
    #@show kls = ((1.+τx)*(1.- βhat*(1.-δ))/(βhat*θ))^(1/(θ-1.))*z
    #A = (z/kls)^(1.-θ)-(1+gz)*(1.+gn)+1.-δ
    #B = (1.-τl)*(1.-θ)*kls^θ*z^(1.- θ)/ψ
    #@show ks = (B+g)/(A+B/kls)
    return kss,lss,css,xss,f
end

"""
First order conditions that determine labor to be linearized

u_c/u_l = (1-τ)F_l
"""
function mris(q::Array{Float64},params::Array{Float64},tol = 1e-10,maxn = 20)
    k,kp,l,z,τl,τx,g = q
    gn,gz,β,δ,ψ,σ,θ = params
    c = k^θ*(z*l)^(1.0 - θ) - (1.0 + gz)*(1.0 + gn)*kp + (1 - δ)*k - g 
    eqn1 = (ψ*c) - (1-l)*(1-τl)*(1-θ)*(k/l)^θ*(z)^(1.0 - θ)
    return eqn1
end

"""
Production function to be linearized
"""
function production(q::Array{Float64},params::Array{Float64})
    k,h,z = q
    gn,gz,β,δ,ψ,σ,θ = params
    f = k^θ*(h*z)^(1.-θ)
    return k^θ*(h*z)^(1.-θ) 
end


"""
Investment equation to be linearized
"""
function invest(q::Array{Float64},params::Array{Float64})
    k,kp,x = q
    gn,gz,β,δ,ψ,σ,θ = params
    return (1. + gn)*(1. + gz)*kp - (1. - δ)*k - x
end

"""
Euler equation to be linearized

(1+τx)u_c(c,l) = βhat*u_cp(cp,lp)(F_k + (1-δ)*(1+τxp))

inputs
1. q: vector =  (kpp,kp,k,zp,z,τlp,τl,τxp,τx,gp,g)
2. vector containing fixed parameters
"""
function euler_equation(q::Array{Float64},params::Array{Float64},tol = 1e-10,maxn = 20)
    gn,gz,β,δ,ψ,σ,θ = params
    βhat = β*(1.0+gz)^(-σ)
    
    kpp,kp,k = exp(q[1]),exp(q[2]),exp(q[3])
    zp,z =  exp(q[4]),exp(q[5])
    τlp,τl = q[6],q[7]
    τxp,τx = q[8],q[9]
    gp,g = exp(q[10]),exp(q[11])
    
    
    function mri(Kp,K,L,Z,Τl,Τx,G) 
        yy = K^θ*(Z*L)^(1.0 - θ)
        cc = yy - (1.0 + gz)*(1.0 + gn)*Kp + (1 - δ)*K - G
        return ψ*cc*L/yy - (1-Τl)*(1-θ)*(1.-L)
    end

    #Solve the static condition for labor
    l,lp = 1./(1.+0.75*ψ/(1.-τl)/(1.-θ)),1./(1.+0.75*ψ/(1.-τlp)/(1.-θ))
    for i = 1:6
        l += -mri(kp,k,l,z,τl,τx,g)/Calculus.derivative(t -> mri(kp,k,t,z,τl,τx,g),l)
        lp += -mri(kpp,kp,lp,zp,τlp,τxp,gp)/Calculus.derivative(t -> mri(kpp,kp,t,zp,τlp,τxp,gp),lp)
    end

    y = k^θ*(z*l)^(1.0 - θ)
    yp = kp^θ*(zp*lp)^(1.0 - θ)
    c = y - (1.0 + gz)*(1.0 + gn)*kp + (1 - δ)*k - g
    cp = yp - (1.0 + gz)*(1.0 + gn)*kpp + (1 - δ)*kp - gp
    
    return (1+τx)*c^(-σ)*(1.-l)^(ψ*(1.-σ)) - βhat*cp^(-σ)*(1.-lp)^(ψ*(1.-σ))*
        (θ*yp/kp + (1.-δ)*(1.+τxp))
end


"""
Inputs:
Theta: parameter vector to be estimated in MLE
model_pars: parameters of the individual and economywide model
exog_ss: exogenous steady state values zss,τlss,τxss,gss
observables: series for Yt = log y_t, log x_t, log l_t, log g_t

Output:
L: value of libelihood for Theta
Sbar: ss values of exo variables
P0: Constant matrix to be used in tauchen
P: coefficient matrix to be used in tauchen
Q: covariance matrix to be used in tauchen
A: A matrix above
B: B matrix above
C: C matrix above
model_pars: parameters of the model to be used later
"""
function mleq(
        Theta::Vector,
        params::Array{Float64},
        observables::Array{Float64})
    
    gn,gz,β,δ,ψ,σ,θ = params
    βhat = β*(1.0+gz)^(-σ)

    #Start up matrices
    P = 0.995*eye(4)
    Sbar = [log(1.);0.05;0.0;log(0.07)]
    
    Q = zeros(4,4)
    D = zeros(4,4)
    R = zeros(4,4)
    
    
    ##### matrices to be estimated ####
    Sbar[1] = Theta[1]
    Sbar[2] = Theta[2]
    Sbar[3] = Theta[3]
    Sbar[4] = Theta[4]
    P[1,1] = Theta[5]
    P[2,1] = Theta[6]
    P[3,1] = Theta[7]
    P[4,1] = Theta[8]
    P[1,2] = Theta[9]
    P[2,2] = Theta[10]
    P[3,2] = Theta[11]
    P[4,2] = Theta[12]
    P[1,3] = Theta[13]
    P[2,3] = Theta[14]
    P[3,3] = Theta[15]
    P[4,3] = Theta[16]
    P[1,4] = Theta[17]
    P[2,4] = Theta[18]
    P[3,4] = Theta[19]
    P[4,4] = Theta[20]
    Q[1,1] = Theta[21]
    Q[2,1] = Theta[22]
    Q[3,1] = Theta[23]
    Q[4,1] = Theta[24]
    Q[2,2] = Theta[25]
    Q[3,2] = Theta[26]
    Q[4,2] = Theta[27]
    Q[3,3] = Theta[28]
    Q[4,3] = Theta[29]
    Q[4,4] = Theta[30]
    
    P0 = (eye(4) - P)*Sbar
    penalty = 500000.*max(maximum(abs.(eig(P)[1]))-.995,0.)^2.
    
    kss,lss,css,xss,yss = equilibrium_ss([2.0;0.5],Sbar,params)
    
    #tem = (eye(4)-P)\P0
    zss,τlss,τxss,gss = exp(Sbar[1]),Sbar[2],Sbar[3],exp(Sbar[4])
    rss = θ*yss/kss
    
    X0 = [log(kss);log(zss);τlss;τxss;log(gss);1.0]
    Y0 = [log(yss);log(xss);log(lss);log(gss)] 
    
    ######################################
    #linearize some equations around SS
    ######################################
    #MRIS
    #log(l) = ϕlk*log(k) + ϕlz*log(z) + ϕll*τl + ϕlg*log(g) + ϕlkp*log(kp)
    ys = [kss,kss,lss,zss,τlss,τxss,gss]
    ys_mul = [kss,kss,lss,zss,1.0,1.0,gss] #1.0s for taxes bc approximation around level for taxes
    ϕl_b = ys_mul.*Calculus.gradient(t -> mris(t,params),ys)
    ϕlh = -ϕl_b[3]
    ϕ_l = ϕl_b/(ϕlh) #normalize so that we can write as log(l)
    ϕlk,ϕlkp,ϕlz,ϕll,ϕlx,ϕlg = ϕ_l[1],ϕ_l[2],ϕ_l[4],ϕ_l[5],ϕ_l[6],ϕ_l[7]
    
    #Production
    #log(y) = ϕyk*log(k) + ϕyz*log(z) + ϕyl*τl + ϕyg*log(g) + ϕykp*log(kp)
    ys = [kss,lss,zss]
    ϕy_b = ys.*Calculus.gradient(t -> production(t,params),ys)
    ϕy = ϕy_b/production(ys,params)
    ϕyk = ϕy[1] + ϕy[2]*ϕlk
    ϕyz = ϕy[3] + ϕy[2]*ϕlz
    ϕyl = ϕy[2]*ϕll
    ϕyg = ϕy[2]*ϕlg
    ϕykp = ϕy[2]*ϕlkp
    ϕyx = 0.0
    
    
    #Investment
    #log(x) = ϕxkp*log(kp) + ϕxk*log(k) 
    ys = [kss,kss,xss]
    ϕx_b = ys.*Calculus.gradient(t -> invest(t,params),ys)
    ϕx = ϕx_b/(-ϕx_b[3]) 
    ϕxk,ϕxkp = ϕx[1],ϕx[2]
    ϕxz = 0.0
    ϕxl = 0.0
    ϕxx = 0.0
    ϕxg = 0.0
 
    ######################################
    #Solve for individual law of motion using Euler equation
    #####################################  
    Z = [log(kss);log(kss);log(kss);log(zss);log(zss);τlss;τlss;τxss;τxss;log(gss);log(gss)]
    del = max.(abs.(Z)*1e-5,1e-8) 
    dR = zeros(11)
    for i = 1:11
        Zp = copy(Z)
        Zm = copy(Z)
        Zp[i] = Z[i]+del[i]
        Zm[i] = Z[i]-del[i]
        euler_equation(Zp,params)
        euler_equation(Zm,params)
        dR[i] = (euler_equation(Zp,params) - euler_equation(Zm,params))/(2.*del[i])
    end
    
    a0 = dR[1]
    a1 = dR[2]
    a2 = dR[3]
    b0 = collect(dR[4:2:11])
    b1 = collect(dR[5:2:11])
    
    roots= [(-a1 - sqrt(a1^2 - 4.0*a0*a2))/(2.0*a0),(-a1 + sqrt(a1^2 - 4.0*a0*a2))/(2.0*a0)]
    index = find(x -> abs(x)<1.0,roots) 
    γk = roots[index...]
    
    γ = -((a0*γk+a1)*eye(4)+a0*P')\(b0'*P + b1')'
    γz,γl,γx,γg = γ[1],γ[2],γ[3],γ[4]
    γ0 = (1.0-γk)*log(kss) - γz*log(zss) - γl*τlss - γx*τxss - γg*log(gss)
    Γ = [γk;γ;γ0] 
    
    ######################################
    # Write linear model
    ######################################
    A = [γk                γz γl γx γg      γ0;
         [0.0;0.0;0.0;0.0] P                P0;
        0.0               0.0 0.0 0.0 0.0  1.0]
    
    B = [0.0 0.0 0.0 0.0;
          Q             ;
         0.0 0.0 0.0 0.0]
    Ctemp = [[ϕyk ϕyz ϕyl ϕyx ϕyg] + ϕykp*Γ[1:5]';
         [ϕxk ϕxz ϕxl ϕxx ϕxg] + ϕxkp*Γ[1:5]';
         [ϕlk ϕlz ϕll ϕlx ϕlg] + ϕlkp*Γ[1:5]';
         0.0 0.0 0.0 0.0 1.0]
    
    ϕ0 = Y0 - Ctemp*X0[1:5]
    C = [Ctemp ϕ0]

    ######################################
    # Specify observables 
    ######################################
    observables = observables[:,2:5]
    T = size(observables,1)
    trend = [(1.+gz).^collect(0:T-1) (1.+gz).^collect(0:T-1) ones(T) (1.+gz).^collect(0:T-1)]  
    Y = log.(observables) - log.(trend)   

    #Might not be zero in other models
    D = zeros(4,4)
    R = zeros(4,4)

    ##transforming state space for MLE
    Ȳ       = Y[2:T,:]
    Tm       = T-1
    C̄       = C*A-D*C
    R̄       = R+C*B*B'*C'
    
    ##Initialize Kalman
    k = zeros(6,4)
    Σ0 = zeros(6,6)
    L = 0.0
    Ω = zeros(4,4)

    innov0 = Ȳ[1,:] - C̄*X0 
    Ωdet = det(Ω)
    sum1 =0.0
    for t=2:Tm
        #covariance of u
        Ω = C̄*Σ0*C̄' + R̄  
        Ωdet  = det(Ω)
        if Ωdet <0.0
            error("Ω negative - cannot take lof of it")
        end
        #kalman gain
        k = (B*B'*C' + A*Σ0*C̄')*inv(Ω) 
        
        ###update X and Σ
        Xnew = A*X0 + k*innov0
        
        #innovation
        innovnew = Ȳ[t,:] - C̄*Xnew         

        Σnew = A*Σ0*A' + B*B' - k*(C̄*Σ0*A' + C*B*B')
        
        sum1 += innov0*innov0'/Tm
        ##Need to add the last innovation using updated innov
        if t == Tm
            sum1 = sum1 + innovnew*innovnew'/Tm
        end
        
        X0,Σ0,innov0 = Xnew,Σnew,innovnew
    end

    L = 0.5*(Tm*(log(Ωdet) + trace(Ω \ sum1)) + penalty)
    
    return L,Sbar,P0,P,Q,A,B,C
end


"""
Inputs:
θ: solution parameter from mle in order to get mle parameters given solution
initial_t: this is the period t that corresponds to SS t
params: model parameters
exog_params: exogenous shocks in steady state
data: data used in mle

Output:
z: matrix with 4 columns, one for each wedge: log z_t, τl_t, τx_t, log g_t

"""
function log_lin_wedges(
    theta::Vector, #comes from maximum likelihood estimation
    initial_t::Int64, #steady state t
    params::Array{Float64}, 
    data::Array{Float64},
    frequency::Int64 = 1) # 1 = quarterly, 0 = annual 
    #Estimated parameters
    #t    = data[1:40,1]
    ZVAR = data[1:end,2:5]
    if size(data)[2] == 6
        KBEA = data[1:end,6]
    end

    if frequency == 1
        L,Sbar,P0,P,Q,A,B,C = mleq(theta,params,data)
    else
        L,Sbar,P0,P,Q,A,B,C = mlea(theta,params,data)
    end

    #model parameters
    gn,gz,β,δ,ψ,σ,θ = params


    #Length of data
    Y0      = initial_t

    #get exogenous variables from estimated parameters
    z       = exp(Sbar[1])
    τl    = Sbar[2]
    τx    = Sbar[3]
    g       = exp(Sbar[4])


    #get steady state equilibrium conditions
    #k,l,c,x,y = equilibrium_ss(2.5,0.5,Sbar,params)
    k,l,c,x,y = equilibrium_ss([10.0;0.5],Sbar,params)
    βhat    = β*(1+gz)^(-σ)

    lk      = log(k)
    lc      = log(c)
    ll      = log(l)
    ly      = log(y)
    lx      = log(x)
    lg      = log(g)
    lz      = log(z)
    T,_     = size(ZVAR)
    
    #Detrend data and take logs
    trend = [(1.+gz).^collect(0:T-1) (1.+gz).^collect(0:T-1) ones(T) (1.+gz).^collect(0:T-1)]  
    Y = log.(ZVAR) - log.(trend)
    
    #Prepare data on output, investment, hours,government exp and capital stock
    #log of detrended observables
    if size(data)[2] == 6
        lkbea   = log.(KBEA) - log.((1+gz).^collect(0:T-1));
    end
    lyt     = Y[:,1];
    lxt     = Y[:,2];
    llt     = Y[:,3];
    lgt     = Y[:,4];

    #Need variable analogs from model, individual capital, and capital stock
    lkt = zeros(T+1)
    lktp = zeros(T)
    Kt = zeros(T+1)
    Ktp = zeros(T)


    #Initialize at year 1929
    lkt[Y0]= lk
    if size(data)[2] == 6
        Kt[Y0] = exp(lkbea[Y0])
    else
        Kt[Y0] = exp(lk)
    end

    for i=Y0:T;
      lktp[i]  = lk+((1-δ)*(lkt[i]-lk)+x/k*(lxt[i]-lx))/(1+gz)/(1+gn);
      lkt[i+1] = lktp[i];
      Ktp[i]   = ((1-δ)*Kt[i]+exp(lxt[i]))/(1+gz)/(1+gn)
      Kt[i+1]  = Ktp[i]
    end;
    #@show Kt
    #@show lkt
    
    #for i=Y0-1:-1:1;
    #  lktp[i]  = lkt[i+1]
    #  lkt[i]   = lk+((1+gz)*(1+gn)*(lktp[i]-lk)-x/k*(lxt[i]-lx))/(1-δ)
    #  Ktp[i]   = Kt[i+1]
    #  Kt[i]    = ((1+gz)*(1+gn)*Ktp[i]-exp(lxt[i]))/(1-δ)
    #end

    lkt      = lkt[1:T];
    Kt       = Kt[1:T];
    
    #From individual and aggregate capital, can generate other variables
    lct,lzt,tault,tauxt,tauxchk = zeros(T),zeros(T),zeros(T),zeros(T),zeros(T)
    Ct,Zt,Ztbea,Tault = zeros(T),zeros(T),zeros(T),zeros(T)

    #start_T = 81
    #end_T = 81+29
    for t = 1:T
        lct[t] = lc + (y*(lyt[t] - ly) - x*(lxt[t] - lx) - g*(lgt[t] - lg))/c
        lzt[t] = lz + (lyt[t] - ly - θ*(lkt[t] - lk))/(1-θ) - llt[t] + ll
        tault[t] = τl + (1-τl)*(lyt[t] - ly - lct[t] + lc - 1.0/(1.0 - l)*(llt[t] - ll))
        tauxt[t] = (lxt[t] - C[2,1]*lkt[t] - C[2,2]*lzt[t] -C[2,3]*tault[t] - C[2,5]*lgt[t] - C[2,6])/C[2,4]
        tauxchk[t]  = (lyt[t] - C[1,1]*lkt[t] - C[1,2]*lzt[t] - C[1,3]*tault[t] - C[1,5]*lgt[t] - C[1,6])/C[1,4]
        Ct[t]       = exp(lyt[t])-exp(lxt[t])-exp(lgt[t])       
        Zt[t] = (exp(lyt[t])/(Kt[t]^θ*exp(llt[t])^(1.0 - θ)))^(1.0/(1.0 - θ))
        Tault[t]    = 1.0 - ψ/(1.0 - θ)*(Ct[t]/exp(lyt[t]))*(exp(llt[t])/(1.0 - exp(llt[t])))
    end
    
    return P0,P,Q,lyt, [log.(Zt) Tault tauxt log.(exp.(lgt))]
end
