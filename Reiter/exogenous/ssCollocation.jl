using LinearAlgebra
using Parameters
using IterativeSolvers
using ForwardDiff
using Plots
using Arpack
################################### Model types #########################

struct ReiterParameters{R <: Real}
    β::R
    d::R
    A::R
    B_::R
    γ::R
    α::R
    T::R
    ζ::R
    L::R
    homeY
end

struct ReiterCollocation{R <: Real,I <: Integer}
    aGrid::Array{R,1}
    na::I
end

struct ReiterMarkovChain{R <: Real,I <: Integer}
    states::Array{R,1}
    ns::I
    stateID::Array{I,1}
    Π::Array{R,2}
end

struct ReiterDistribution{R <: Real,I <: Integer}
    DistributionSize::I
    DistributionAssetGrid::Array{R,1}
end

#struct ReiterRE{R <: Real,I <: Integer}
    
    

struct ReiterMethod{R <: Real,I <: Integer}
    Guess::Array{R,1}
    GuessMatrix::Array{R,2}
    Parameters::ReiterParameters{R}
    Collocation::ReiterCollocation{R,I}
    MarkovChain::ReiterMarkovChain{R,I}
    Distribution::ReiterDistribution{R,I}
end

function Reiters(
    InitialR::R,
    InitialL::R,
    β::R = 0.97,
    d::R = 0.04,
    A::R = 1.0,
    B_::R = 0.0,
    γ::R = 1.0/2.9,
    α::R = 0.3,
    Π01 = 0.5,
    Π10 = 0.038,
    T::R = 1.0,
    ζ::R = 10000000000000.0,
    homeY::R = 0.07,
    na::I = 40,
    aMax::R = 30.0,
    gS::R = 1.0,
    bS::R = 0.0,
    ns::I = 2,
    DistributionSize::I = 300) where{R <: Real,I <: Integer}

    ################## Collocation pieces
    function grid_fun(a_min,a_max,na, pexp)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
        return grid
    end
    aGrid = grid_fun(0.0,aMax,na,4.0)
    #aGrid = collect(range(0.0,stop = aMax,length = na))
    Collocation = ReiterCollocation(aGrid,na)

    ################### Distribution pieces
    DistributionAssetGrid = collect(range(aGrid[1],stop = aGrid[end],length = DistributionSize))
    Distribution = ReiterDistribution(DistributionSize,DistributionAssetGrid)

    ###Exogenous states and Markov chain
    IndividualStates = [gS;bS]
    TransitionMatrix = [0.9 0.1; 0.5 0.5]
    MarkovChain = ReiterMarkovChain(IndividualStates,ns,[1;2],TransitionMatrix)

    ################### Final model pieces
    r = InitialR
    H = InitialL
    w = (1-α)*T*((T*α)/(r+d))^(α/(1-α))
    Guess = zeros(na*ns)
    for (si,ϵ) in enumerate(IndividualStates)
        for (ki,k) in enumerate(aGrid) 
            n = (si-1)*na + ki
            si == 1 ? ap = (1.0 + r)*k + w*ϵ*0.5  - 0.7 : ap = (1.0 + r)*k + homeY  - 0.5 
            Guess[n] = 0.985*ap
        end
    end
    GuessMatrix = reshape(Guess,na,ns)
    L = 1.0 - Π10/(Π01 - Π10)    
    Parameters = ReiterParameters(β,d,A,B_,γ,α,T,ζ,L,homeY)
    
    return ReiterMethod(Guess,GuessMatrix,Parameters,Collocation,MarkovChain,Distribution)
end

function Residual(
    θ::Array{R,1},
    InterestRate::F,
    ReiterObj::ReiterMethod{F,I}) where{R <: Real,F<:Real,I <: Integer}
    
    #Model parameters
    @unpack β,d,A,B_,γ,α,T,ζ,L,homeY = ReiterObj.Parameters  
    @unpack aGrid,na = ReiterObj.Collocation  
    @unpack states,ns,stateID,Π = ReiterObj.MarkovChain  
    nx = na*ns
    
    #model FiniteElementObj.Parameters
    r = InterestRate
    w = (1.0-α)*T*((T*α)/(r+d))^(α/(1.0-α))
    l,c,uc,ucc,ul,ull = 0.0,0.0,0.0,0.0,0.0,0.0
    lp,cp,ucp,uccp,ulp,ullp = 0.0,0.0,0.0,0.0,0.0,0.0
    
    dResidual = zeros(R,nx,nx)
    Residual = zeros(R,nx)
    np = 0    
    for (s,ϵ) in enumerate(states)
        for (n,a) in enumerate(aGrid)
            s1 = (s-1)*na + n

            #Policy functions
            ap = θ[s1]

            #penalty function
            pen = ζ*min(ap,0.0)^2
            dpen = 2*ζ*min(ap,0.0)

            c = (1.0 + r)*a + ϵ*w*(1.0 - l) + (1.0 - ϵ)*homeY - ap
            uc = 1.0/c
            ucc = -1.0/c^2.0                    
            ∂c∂ai = -1.0

            np = searchsortedlast(aGrid,ap)

            ##Adjust indices if assets fall out of bounds
            (np > 0 && np < na) ? np = np : 
                (np == na) ? np = na-1 : 
                    np = 1 

            ap1,ap2 = aGrid[np],aGrid[np+1]
            basisp1 = (ap2 - ap)/(ap2 - ap1)
            basisp2 = (ap - ap1)/(ap2 - ap1)

            ####### Store derivatives###############
            dbasisp1 = -1.0/(ap2 - ap1)
            dbasisp2 = -dbasisp1                

            tsai = 0.0
            sum1 = 0.0
            for sp = 1:ns
                sp1 = (sp-1)*na + np
                sp2 = (sp-1)*na + np + 1
                ϵp = states[sp]

                #Policy functions
                app = θ[sp1]*basisp1 + θ[sp2]*basisp2
                cp = (1.0 + r)*ap + ϵp*w*(1.0 - lp) + (1.0 - ϵp)*homeY - app
                ucp = 1.0/cp
                uccp = -1.0/cp^2.0                    

                #Need ∂cp∂ai and ∂cp∂aj
                ∂ap∂ai = θ[sp1]*dbasisp1 + θ[sp2]*dbasisp2
                ∂cp∂ai = (1.0 + r) - ∂ap∂ai
                ∂cp∂aj = -1.0

                sum1 += β*(Π[s,sp]*(1.0 + r)*ucp + pen) 

                #summing derivatives with respect to θs_i associated with c(s)
                tsai += β*Π[s,sp]*(1.0 + r)*uccp*∂cp∂ai + dpen
                tsaj = β*Π[s,sp]*(1.0 + r)*uccp*∂cp∂aj

                dResidual[s1,sp1] +=   tsaj * basisp1
                dResidual[s1,sp2] +=   tsaj * basisp2
            end
            ##add the LHS and RHS of euler for each s wrt to θi
            dres =  tsai - ucc*∂c∂ai

            dResidual[s1,s1] += dres

            res = sum1 - uc
            Residual[s1] += res
        end
    end 
    Residual,dResidual 
end


function SolveCollocation(
    guess::Array{R,1},
    InterestRate::R,
    ReiterObj::ReiterMethod,
    maxn::Int64 = 100,
    tol = 1e-9
) where{R <: Real,I <: Integer}

    θ = guess
    #Newton Iteration
    for i = 1:maxn
        Res,dRes = Residual(θ,InterestRate,ReiterObj)
        step = - dRes \ Res
        if LinearAlgebra.norm(step) >1.0
            θ += 1.0/10.0*step
        else
            θ += 1.0/1.0*step
        end
        #@show LinearAlgebra.norm(step)
        if LinearAlgebra.norm(step) < tol
            println("Individual problem converged in ",i," steps")
            return θ
            break
        end
    end
        
    return println("Individual problem Did not converge")
end


function StationaryDistribution(
    InterestRate::R,
    θ::Array{R,1},
    ReiterObj::ReiterMethod{F,I}
) where{R <: Real,I <: Integer,F <: Real}

    @unpack β,d,A,B_,γ,α,T,ζ,L,homeY = ReiterObj.Parameters  
    @unpack aGrid,na = ReiterObj.Collocation  
    @unpack states,ns,stateID,Π = ReiterObj.MarkovChain  
    @unpack DistributionSize, DistributionAssetGrid = ReiterObj.Distribution
    
    nf = ns*DistributionSize
    r = InterestRate
    w = (1.0-α)*T*((T*α)/(r+d))^(α/(1.0-α))
    θ = reshape(θ,na,ns)
    l,lp = 0.0,0.0
    ##initialize
    pdf1 = zeros(nf)
    Qa = zeros(nf,nf)
    #c,ap =zeros(DistributionSize,ns),zeros(DistributionSize,ns)

    for s=1:ns
        for i=1:DistributionSize
            x = DistributionAssetGrid[i] 
            
            ######
            # find each k in dist grid in nodes to use FEM solution
            ######
            n = searchsortedlast(aGrid,x)
            (n > 0 && n < na) ? n = n : 
                (n == na) ? n = na-1 : 
                    n = 1 
            x1,x2 = aGrid[n],aGrid[n+1]
            basis1 = (x2 - x)/(x2 - x1)
            basis2 = (x - x1)/(x2 - x1)
            ap  = basis1*θ[n,s] + basis2*θ[n+1,s]            
            
            ######
            # Find in dist grid where policy function is
            ######            
            n = searchsortedlast(DistributionAssetGrid,ap)
            if (n > 0 && n < DistributionSize)
                aph,apl = DistributionAssetGrid[n+1],DistributionAssetGrid[n]
            end
            ######
            # Build histogram
            ######            
            for si = 1:ns
                aa = (s-1)*DistributionSize + i
                ss = (si-1)*DistributionSize + n
                if n > 0 && n < DistributionSize
                    ω = (ap - apl)/(aph - apl)
                    Qa[aa,ss+1] += Π[s,si]*ω
                    Qa[aa,ss]  += Π[s,si]*(1.0 - ω)
                elseif n == 0
                    ω = 1.0
                    Qa[aa,ss+1] += Π[s,si]*ω
                else
                    ω = 1.0
                    Qa[aa,ss] += Π[s,si]*ω
                end
            end
        end
    end

    for i = 1:nf
        for j = 1:nf
            (Qa[i,j] == 0.0) ? Qa[i,j] = 0.00000000000001 : Qa[i,j] = Qa[i,j]
        end
    end   
    
    #Get the eigen vector of unity eigenvalue by power method
    λ, x = powm!(transpose(Qa), fill(1.0,(nf,)), maxiter = 1000,tol = 1e-10)
    #eigevl,eigenv = eigen(Qa)
    #@show real(eigenv[1,:])
    #@show real(eigevl)
    #@show eigenv = real(squeeze(eigenv,2))
    #@show λ
    #renormalize eigen vector so it adds up to one by state
    for i = 1:nf
        pdf1[i] = 1.0/sum(x) * x[i]
    end

    EA = 0.0
    for s = 1:ns
        for ki = 1:DistributionSize
            i = (s-1)*DistributionSize + ki
            EA += pdf1[i]*DistributionAssetGrid[ki]            
        end
    end

    EA,pdf1
end


function equilibrium(
    ReiterObj::ReiterMethod{R,I},
    tol = 1e-10,maxn = 100
) where{R <: Real,I <: Integer}

    @unpack β,d,A,B_,γ,α,T,ζ,L,homeY = ReiterObj.Parameters  
    @unpack aGrid,na = ReiterObj.Collocation  
    @unpack states,ns,stateID,Π = ReiterObj.MarkovChain  
    @unpack DistributionSize, DistributionAssetGrid = ReiterObj.Distribution

    ull,ucc,uc,∂l∂L = 0.0,0.0,0.0,0.0
    
    #Bisection method for equilibrium
    #store some policies
    cpol = zeros(R,DistributionSize,ns)
    lpol = zeros(R,DistributionSize,ns)
    appol = zeros(R,DistributionSize,ns)

    AssetDistribution = zeros(DistributionSize*ns)
    EA = 0.0
    
    ###Start Bisection
    #L = 1.0  #labor demand guess
    K = 0.0
    θeq = ReiterObj.Guess

    r0 = 0.02
    uir,lir = 0.04, 0.001
    for kit = 1:maxn            
        θeq = SolveCollocation(θeq,r0,ReiterObj)
        EA,AssetDistribution = StationaryDistribution(r0,θeq,ReiterObj)

        @show EA
        ### Implicit interest rate
        rd = α*T*EA^(α - 1.0)*L^(1.0-α) - d
        #@show r0
        ### narrow interval by updating upper and lower bounds on 
        ### interval to search new root
        if (rd > r0)
            uir = min(uir,rd)
            lir = max(lir,r0)
            r0 = 1.0/2.0*(uir + lir)
        else
            lir = max(lir,rd)
            uir = min(uir,r0)
            r0 = 1.0/2.0*(uir + lir)
        end
        #@show r
        #@show rd
        if abs(r0 - rd) < 1e-14
            println("Capital markets clear at interest rate: ",r0)
            return θeq,EA,r0,AssetDistribution
            break
        end
        #r0 = r
    end
    
    return println("Markets did not clear")
end


function reiter_eqn_builder(
    y::Array{R,1},
    ReiterObj::ReiterMethod{F,I}) where{R <: Real,I <: Integer,F <: Real}
    
    @unpack β,d,A,B_,γ,α,T,ζ,L,homeY = ReiterObj.Parameters  
    @unpack aGrid,na = ReiterObj.Collocation  
    @unpack states,ns,stateID,Π = ReiterObj.MarkovChain  
    @unpack DistributionSize, DistributionAssetGrid = ReiterObj.Distribution
        
    nx = DistributionSize
    nf = ns*nx
    nfa = ns*na

    ρz,σz = 0.9,0.01
    #############################
    #Unknowns to be determined
    #############################
    #y = (θ), x = (Φ,z)
    Lp = L
    Ny,Nx,Nϵ,Nη = nfa,nf+1,1,nfa
    VarId = Dict{Symbol,UnitRange{I}}()
    VarId[:yp] = 1:Ny
    VarId[:y]  = Ny+1:2*Ny
    VarId[:xp] = 2*Ny+1:2*Ny+Nx
    VarId[:x]  = 2*Ny+Nx+1:2*Ny+2*Nx
    VarId[:η]  = 2*Ny+2*Nx+1:2*Ny+2*Nx+nfa
    ωi  = 2*Ny+2*Nx+nfa+1
    
    θp,θ = y[VarId[:yp]], y[VarId[:y]]
    λp,zp = reshape(y[VarId[:xp]][1:end-1],nx,ns),y[VarId[:xp]][end]
    λ,z = reshape(y[VarId[:x]][1:end-1],nx,ns),y[VarId[:x]][end]
    ηe = y[VarId[:η]]
    ωz = y[ωi]

    K = 0.0
    for s = 1:ns
        for i = 1:nx
            K += λ[i,s]*DistributionAssetGrid[i]
        end
    end
    Kp = 0.0
    for s = 1:ns
        for i = 1:nx
            Kp += λp[i,s]*DistributionAssetGrid[i]
        end
    end
    #@show K
    #@show Kp
    w = (1.0-α)*exp(z)*(K/L)^α
    r = α*exp(z)*(K/L)^(α-1) - d
    wp = (1.0-α)*exp(zp - σz*ωz)*(Kp/Lp)^α
    rp = α*exp(zp - σz*ωz)*(Kp/Lp)^(α-1) - d    

    ###################################################
    f = zeros(R,Nx+Ny)
    #@show nfa+nf+1
    #@show div(length(y)-1,2)
    if length(f) != div(length(y)-1-nfa,2)
        error("# of equations not equal to # of variables")
    end
    ##################################
    """
    λp = ∑λ              nf equations
    EE[θ]                nfa equations
    zp - ρ_z*z - ρ_ϵ*ϵ   1 equation
    yp: nfa  variables
    xp: nf+1 variables
    ω: 1 variable
    """
    his,his_rhs = zeros(R,nf),zeros(R,nf)    ## Fill this
    EE = zeros(R,nfa)                        ## Fill this
    shocks =  zp - ρz*z - σz*ωz
    ##########################################################
    # For the pieces from distribution
    ##########################################################
    
    for s=1:ns
        for i=1:nx
            ii = (s-1)*nx + i
            a = DistributionAssetGrid[i] 
            
            ######
            # find each k in dist grid in nodes to use FEM solution
            ######
            n = searchsortedlast(aGrid,a)
            (n > 0 && n < na) ? n = n : 
                (n == na) ? n = na-1 : 
                     n = 1
            s1 = (s-1)*na + n
            s2 = (s-1)*na + n + 1
            a1,a2 = aGrid[n],aGrid[n+1]
            basis1 = (a2 - a)/(a2 - a1)
            basis2 = (a - a1)/(a2 - a1)
            ap  = basis1*θ[s1] + basis2*θ[s2]            
            
            ######
            # Find in dist grid where policy function is
            ######            
            n = searchsortedlast(DistributionAssetGrid,ap)
            if (n > 0 && n < nx)
                  aph,apl = DistributionAssetGrid[n+1],DistributionAssetGrid[n]
            end
            ######
            # Build histogram
            ######
            for sp = 1:ns
                #aa = (s-1)*nx + i
                ss = (sp-1)*nx + n
                if n > 0 && n < nx
                    ω = 1.0 - (ap - apl)/(aph - apl)
                    his_rhs[ss+1] += Π[s,sp] * (1.0 - ω) * λ[i,s]
                    his_rhs[ss]  += Π[s,sp] * ω * λ[i,s]
                elseif n == 0
                    ω = 1.0
                    his_rhs[ss+1] += Π[s,sp] * ω * λ[i,s]
                else
                    ω = 1.0
                    his_rhs[ss] += Π[s,sp] * ω * λ[i,s]
                end
            end 
        end
    end
    for s = 1:ns
        for i = 1:nx
            ii = (s-1)*nx + i
            his[ii] = λp[i,s] - his_rhs[ii]
        end
    end
    
    ##########################################################
    # For the pieces from FEM
    ##########################################################
    c,l,uc,ucc,ul,ull = 0.0,0.0,0.0,0.0,0.0,0.0
    cp,lp,ucp,uccp,ulp,ullp = 0.0,0.0,0.0,0.0,0.0,0.0
    for s=1:ns
        ϵ = states[s]
        for i=1:na
            s1 = (s-1)*na + i
            a = aGrid[i]
            
            #Policy functions
            ap = θ[s1]

            c = (1.0 + r)*a + ϵ*w*(1.0 - l) + (1.0 - ϵ)*homeY - ap
            uc = 1.0/c
            ucc = -1.0/c^2.0                    
            
            #xp = a
            np = searchsortedlast(aGrid,ap)
            ##Adjust indices if assets fall out of bounds
            (np > 0 && np < na) ? np = np : 
                (np == na) ? np = na-1 : 
                    np = 1 
            
            ap1,ap2 = aGrid[np],aGrid[np+1]
            basisp1 = (ap2 - ap)/(ap2 - ap1)
            basisp2 = (ap - ap1)/(ap2 - ap1)
            

            ee_rhs = 0.0
            for sp = 1:ns
                sp1 = (sp-1)*na + np
                sp2 = (sp-1)*na + np + 1
                ϵp = states[sp]

                #Policy
                app = θp[sp1]*basisp1 + θp[sp2]*basisp2

                cp = (1.0 + rp)*ap + ϵp*wp*(1.0 - lp) + (1.0 - ϵp)*homeY - app
                ucp = 1.0/cp
                uccp = -1.0/cp^2.0                    

                ###Euler RHS
                ee_rhs += β*Π[s,sp]*(1.0 + rp)*ucp   
            end

            res = uc - ee_rhs - ηe[s1]
            EE[s1] = res 
        end
    end

    f[1:nf] = his
    f[nf+1] = shocks
    f[nf+2:nf+1+nfa] = EE
 

    return f
end
