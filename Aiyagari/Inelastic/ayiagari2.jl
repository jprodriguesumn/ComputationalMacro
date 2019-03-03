using LinearAlgebra
using Parameters
using IterativeSolvers
using Pkg
using Plots
using FastGaussQuadrature
using BenchmarkTools
using ForwardDiff
using DelimitedFiles

################################### Model types #########################

struct ModelParameters{R <: Real}
    β::R
    d::R
    A::R
    B_::R
    γc::R
    γl::R
    α::R
    T::R
    ζ::R
end

struct ModelFiniteElement{R <: Real,I <: Integer}
    nodes::Array{R,1}
    na::I
    m::I
    wx::Array{R,1}
    ax::Array{R,1}
end

struct ModelMarkovChain{R <: Real,I <: Integer}
    states::Array{R,1}
    ns::I
    stateID::Array{I,1}
    Π::Array{R,2}
end

struct ModelDistribution{R <: Real,I <: Integer}
    DistributionSize::I
    DistributionAssetGrid::Array{R,1}
end

struct AyiagariMarkovianFiniteElement{R <: Real,I <: Integer}
    Guess::Array{R,1}
    GuessMatrix::Array{R,2}
    LaborSupply::R
    Parameters::ModelParameters{R}
    FiniteElement::ModelFiniteElement{R,I}
    MarkovChain::ModelMarkovChain{R,I}
    Distribution::ModelDistribution{R,I}
end

"""
Inputs:
nodes: grid on assets


The weighted residual equation and its jacobian

inputs:
m: quad nodes for integration
ns: number of stochastic states
na: grid size on assets

"""
function WeightedResidual(
    θ::Array{R,1},
    InterestRate::R,
    FiniteElementObj::AyiagariMarkovianFiniteElement{R,I}) where{R <: Real,I <: Integer}
    
    #Model parameters
    @unpack β,d,A,B_,γc,γl,α,T,ζ = FiniteElementObj.Parameters  
    @unpack nodes,na,m,wx,ax = FiniteElementObj.FiniteElement  
    @unpack states,ns,stateID,Π = FiniteElementObj.MarkovChain  
    H = FiniteElementObj.LaborSupply
    ne = na-1
    nx = na*ns

    
    #model FiniteElementObj.Parameters
    r = InterestRate
    #w = (1.0 - α)/H
    w = (1-α)*T*( (r+d)/(T*α) )^(α/(α-1.0))
    #w = (1-α)*A*((A*α*H^(1.0-α))/(r+d))^(α/(1-α))*H^(-α)
    l,lp = 0.0,0.0
    dResidual = zeros(nx,nx)
    Residual = zeros(nx)
    np = 0    
    for s = 1:ns
        ϵ = states[s]
        for n=1:ne
            a1,a2 = nodes[n],nodes[n+1]
            s1 = (s-1)*na + n
            s2 = (s-1)*na + n + 1
            for i=1:m
                #transforming k according to Legendre's rule
                x = (a1 + a2)/2.0 + (a2 - a1)/2.0 * ax[i]
                v = (a2-a1)/2.0*wx[i]

                #Form basis for piecewise function
                basis1 = (a2 - x)/(a2 - a1)
                basis2 = (x - a1)/(a2 - a1)

                #Policy functions
                a = θ[s1]*basis1 + θ[s2]*basis2
                c = (1.0 + r)*x + ϵ*w*(1.0 - l) - a

                #penalty function
                uc = c^(-γc)
                ucc = -γc*c^(-γc - 1.0)                    
                ∂c∂ai = -1.0
                pen = min(a-B_,0.0)^2
                dpen = 2*min(a-B_,0.0)
                               
                np = searchsortedlast(nodes,a)

                ##Adjust indices if assets fall out of bounds
                (np > 0 && np < na) ? np = np : 
                    (np == na) ? np = na-1 : 
                        np = 1 

                ap1,ap2 = nodes[np],nodes[np+1]
                basisp1 = (ap2 - a)/(ap2 - ap1)
                basisp2 = (a - ap1)/(ap2 - ap1)

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
                    ap = θ[sp1]*basisp1 + θ[sp2]*basisp2
                    cp = (1.0 + r)*a + ϵp*w*(1.0 - lp) - ap

                    #agents
                    ucp = cp^(-γc)
                    uccp = -γc*cp^(-γc - 1.0)                    
 
                    
                    #Need ∂cp∂ai and ∂cp∂aj
                    ∂ap∂a = θ[sp1]*dbasisp1 + θ[sp2]*dbasisp2
                    ∂cp∂ai = (1.0 + r) - ∂ap∂a 
                    ∂cp∂aj = -1.0

                    sum1 += β*(Π[s,sp]*(1.0 + r)*ucp + ζ*pen) 
                    #sum1 += β*(Π[s,sp]*(1.0 + r)*ucp) 
                    #sum1 = 0.0
                    
                    #summing derivatives with respect to θs_i associated with c(s)
                    tsai += β*(Π[s,sp]*(1.0 + r)*uccp*∂cp∂ai + ζ*dpen)
                    #tsai += β*(Π[s,sp]*(1.0 + r)*uccp*∂cp∂ai)
                    #tsai = 0.0
                    tsaj = β*Π[s,sp]*(1.0 + r)*uccp*∂cp∂aj
                    #tsaj = 0.0

                    dResidual[s1,sp1] +=  basis1 * v * tsaj * basisp1
                    dResidual[s1,sp2] +=  basis1 * v * tsaj * basisp2
                    dResidual[s2,sp1] +=  basis2 * v * tsaj * basisp1
                    dResidual[s2,sp2] +=  basis2 * v * tsaj * basisp2 
                end
                ##add the LHS and RHS of euler for each s wrt to θi
                dres =  tsai - ucc*∂c∂ai
                
                dResidual[s1,s1] +=  basis1 * v * dres * basis1
                dResidual[s1,s2] +=  basis1 * v * dres * basis2
                dResidual[s2,s1] +=  basis2 * v * dres * basis1
                dResidual[s2,s2] +=  basis2 * v * dres * basis2

                res = sum1 - uc


                Residual[s1] += basis1*v*res
                Residual[s2] += basis2*v*res
            end
        end
    end 
    Residual,dResidual 
end


function SolveFiniteElement(
    InterestRate::R,
    guess::Array{R,1},
    FiniteElementObj::AyiagariMarkovianFiniteElement{R,I},
    maxn::Int64 = 2000,
    tol = 1e-13
) where{R <: Real,I <: Integer}

    θ = guess
    #Newton Iteration
    for i = 1:maxn
        Res,dRes = WeightedResidual(θ,InterestRate,FiniteElementObj)
        step = - dRes \ Res
        if LinearAlgebra.norm(step) >1.0
            θ += 1.0/10.0*step
        else
            θ += 1.0/1.0*step
        end
        @show LinearAlgebra.norm(step)
        if LinearAlgebra.norm(step) < tol
            println("number of newton steps: ",i)
            return θ
            break
        end
    end
        
    return println("Did not converge")
end


function StationaryDistribution(
    InterestRate::R,
    θ::Array{R,1},
    FiniteElementObj::AyiagariMarkovianFiniteElement{R,I}
) where{R <: Real,I <: Integer}
    
    #Model parameters
    @unpack β,d,A,B_,γc,γl,α,T,ζ = FiniteElementObj.Parameters  
    @unpack nodes,na,m,wx,ax = FiniteElementObj.FiniteElement  
    @unpack states,ns,stateID,Π = FiniteElementObj.MarkovChain
    NumberOfHouseholds = FiniteElementObj.Distribution.DistributionSize
    res = FiniteElementObj.Distribution.DistributionAssetGrid
    ne = na-1
    nx = na*ns
    nf = NumberOfHouseholds*ns
    
    r = InterestRate
    w = (1-α)*T*( (r+d)/(T*α) )^(α/(α-1.0))
    θ = reshape(θ,na,ns)
    
    ##initialize
    pdf1 = zeros(nf)
    Qa = zeros(nf,nf)
    c,ap =zeros(NumberOfHouseholds,ns),zeros(NumberOfHouseholds,ns)
    
    for s=1:ns
        for i=1:NumberOfHouseholds
            x = res[i] 
            
            ######
            # find each k in dist grid in nodes to use FEM solution
            ######
            n = searchsortedlast(nodes,x)
            (n > 0 && n < na) ? n = n : 
                (n == na) ? n = na-1 : 
                    n = 1 
            x1,x2 = nodes[n],nodes[n+1]
            basis1 = (x2 - x)/(x2 - x1)
            basis2 = (x - x1)/(x2 - x1)
            ap[i,s]  = basis1*θ[n,s] + basis2*θ[n+1,s]
            #c[i,s] = (1.0+r)*x + w*states[s] - ap[i,s] 
            #z[i,s] = (1.0+r)*x + w*states[s]
            
            
            ######
            # Find in dist grid where policy function is
            ######            
            n = searchsortedlast(res,ap[i,s])
            
            ######
            # Build histogram
            ######            
            for si = 1:ns
                aa = (s-1)*NumberOfHouseholds + i
                ss = (si-1)*NumberOfHouseholds + n
                if n > 0 && n < NumberOfHouseholds
                    ω = (ap[i,s] - res[n])/(res[n+1] - res[n])
                    Qa[aa,ss+1] += Π[s,si]*ω
                    Qa[aa,ss]  += Π[s,si]*(1.0 - ω)
                elseif n == 0
                    ω = 1.0
                    Qa[aa,ss+1] += Π[s,si]
                else
                    ω = 1.0
                    Qa[aa,ss] += Π[s,si]
                end
            end
        end
    end

    for i = 1:nf
        for j = 1:nf
            (Qa[i,j] == 0.0) ? Qa[i,j] = 0.0000000000000000001 : Qa[i,j] = Qa[i,j]
        end
    end

    #xx = (eye(Qa) - Qa + ones(Qa))'\ones(size(Qa,1))
    QQ = zeros(nf,nf)
    tot = 0.0
    for i = 1:nf
        tot = sum(Qa[i,:])
        for j = 1:nf
            QQ[i,j] = 1.0/tot * Qa[i,j]
        end
    end
    
    #Get the eigen vector of unity eigenvalue by power method
    λ, x = IterativeSolvers.powm!(transpose(Qa), rand(nf), maxiter = 1000,tol = 1e-10)

    #@show xx-x
    #renormalize eigen vector so it adds up to one by state
    for i = 1:nf
        pdf1[i] = 1.0/sum(x) * x[i]
    end

    EA = 0.0
    #EA2 = 0.0
    for s = 1:ns 
        for ki = 1:NumberOfHouseholds
            i = (s-1)*NumberOfHouseholds + ki
            EA += pdf1[i]*res[ki]
        end
    end
    
    res,EA,pdf1
end


function equilibrium(
    FiniteElementObj::AyiagariMarkovianFiniteElement{R,I},
    tol = 1e-10,maxn = 100
) where{R <: Real,I <: Integer}
    
    @unpack β,d,A,B_,γc,γl,α,T,ζ = FiniteElementObj.Parameters  
    na = FiniteElementObj.FiniteElement.na
    ns = FiniteElementObj.MarkovChain.ns
    H = FiniteElementObj.LaborSupply
    H = 0.5
    #@unpack nodes,na,m,wx,ax = FiniteElementObj.FiniteElement  
    #@unpack states,ns,stateID,Π = FiniteElementObj.MarkovChain  
    #ns = FiniteElementObj.NumberOfIndividualStates
    #nx = 20
    
    #Bisection method for equilibrium
    cm_ir = (1.0/β-1.0) #complete markets interest rate

    #Demand = zeros(nx)
    #Supply = zeros(nx)
    θ_eq = zeros(ns*na) 
    cap_eq = 0.0
    EA = 0.0
    #_,EA,_ = StationaryDistribution(r0,θ_eq,DistributionSize)
    Residual, dResidual = zeros(na*ns), zeros(na*ns,na*ns)
    
    ###Start Bisection
    #r0 = 0.98 * cm_ir
    up_ir,low_ir = cm_ir,0.1*cm_ir #upper and lower bound on bisection
    θ_eq = FiniteElementObj.Guess
    r0 = low_ir
    for i = 1:maxn
        dResidual .= 0.0
        Residual .= 0.0        
        θ_eq = SolveFiniteElement(r0,θ_eq,FiniteElementObj)
        _,EA,_ = StationaryDistribution(r0,θ_eq,FiniteElementObj)
        
        ### Implicit interest rate along
        rd = α*T*EA^(α - 1.0)*H^(1.0-α) - d
                
        
        ### narrow interval by updating upper and lower bounds on 
        ### interval to search new root
        if (rd > r0)
             r = 1.0/2.0*(min(up_ir,rd) + max(low_ir,r0))
             up_ir = min(up_ir,rd)
             low_ir = max(low_ir,r0)
        else
             r = 1.0/2.0*(min(up_ir,r0) + max(low_ir,rd))
             low_ir = max(low_ir,rd)
             up_ir = min(up_ir,r0)
        end
        
        println("ir: ",r0," supply ",EA," demand ",H*((r0 + d)/(T*α))^(1.0/(α - 1.)))
        if abs(r - r0) < 0.0000000000001
            cap_eq = EA
            r0 = r
            break
        end
        r0 = r
    end
    sr = α*d/(r0+d)
    
    return r0,θ_eq,cap_eq,sr 
end 

function PlotEquilibrium(
    FiniteElementObj::AyiagariMarkovianFiniteElement{R,I}
) where{R <: Real,I <: Integer}

    @unpack β,d,A,B_,γc,γl,α,T,ζ = FiniteElementObj.Parameters  
    H = FiniteElementObj.LaborSupply
    #demand(r)= H*( (r + d)/(T*α) )^(1.0/(α - 1.0))
    nx = 25
    Supply = zeros(nx)
    Demand = zeros(nx)
    θ0 = FiniteElementObj.Guess
    #Get a sequence of asset demand and supply for display
    ir  = collect(range(0.3*(1.0/β - 1.0),stop=(1.0/β - 1.0),length=nx))
    for i = 1:nx
        r = ir[i]
        θ0 =  SolveFiniteElement(r,θ0,FiniteElementObj)
        _,EA,_ = StationaryDistribution(r,θ0,FiniteElementObj)
        Supply[i] = EA
        Demand[i] = H*( (r + d)/(T*α) )^(1.0/(α - 1.0))
        #demand(r)
    end

    return Demand,Supply,ir
end
    

"""
Construct and Ayiagari model instace of all parts needed to solve the model
"""
function AyiagariModel(
    InterestRate::R,
    LaborSupply = 1.0,
    β = 0.96,
    d = 0.025,
    A = 1.00,
    B_ = 0.0,
    γc = 1.00,
    γl = 1.0,
    α = 0.36,
    T = 1.0,
    ζ = 10000000000.0,
    GridSize = 40,
    GridMax = 50,
    IndividualStates = [1.0;0.2],
    NumberOfQuadratureNodesPerElement = 2
) where{R <: Real}

    ################## Finite Element pieces
    function grid_fun(a_min,a_max,na, pexp)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
        return grid
    end
    nodes = grid_fun(B_,GridMax,GridSize,4)
    QuadratureAbscissas,QuadratureWeights = gausslegendre(NumberOfQuadratureNodesPerElement)
    NumberOfNodes = GridSize    
    NumberOfElements = NumberOfNodes-1
    NumberOfVertices = 2 
    FiniteElement = ModelFiniteElement(nodes,NumberOfNodes,NumberOfQuadratureNodesPerElement,QuadratureWeights,QuadratureAbscissas)

    ################### Distribution pieces
    DistUpperBound = 70.0
    NumberOfHouseholds = 500
    DistributionAssetGrid = collect(range(nodes[1],stop = DistUpperBound,length = NumberOfHouseholds))
    Distribution = ModelDistribution(NumberOfHouseholds,DistributionAssetGrid)

    ###Exogenous states and Markov chain
    NumberOfIndividualStates = size(IndividualStates,1)
    #TransitionMatrix = [0.94 0.06; 0.91 0.09]
    TransitionMatrix = [0.9 0.1; 0.7 0.3]
    MarkovChain = ModelMarkovChain(IndividualStates,NumberOfIndividualStates,[1;2],TransitionMatrix)
    ################### Final model pieces

    r = InterestRate
    H = LaborSupply
    #@show w = (1.0 - α)/H
    w = (1-α)*T*(T*α/(r+d))^(α/(1-α))
    #@show w = (1-α)*A*((A*α)/(r+d))^(α/(1-α))
    Guess = zeros(NumberOfNodes*NumberOfIndividualStates)
    for j=1:NumberOfIndividualStates
        for i=1:NumberOfNodes
            n = (j-1)*NumberOfNodes + i
            assets = (1.0 + r)*nodes[i] + w*IndividualStates[j]  
            Guess[n] = 0.97* assets 
            #if assets > B_
            #    println("node where assets exceed borrowing constraint: ",j," ",i)
            #end
            #Guess[n] = 19.0/20.0*assets
        end
    end
    GuessMatrix = reshape(Guess,NumberOfNodes,NumberOfIndividualStates)
        
    ################## Maybe elements and element indices
    ElementVertexIndices = ones(Integer,NumberOfVertices,NumberOfElements) #element indices
    ElementVertices = zeros(NumberOfVertices,NumberOfElements)
    for i = 1:NumberOfElements
        ElementVertexIndices[1,i],ElementVertexIndices[2,i] = i,i+1     
        ElementVertices[1,i],ElementVertices[2,i] = nodes[i],nodes[i+1]
    end

    Parameters = ModelParameters(β,d,A,B_,γc,γl,α,T,ζ)
    
    AyiagariMarkovianFiniteElement(Guess,GuessMatrix,H,Parameters,FiniteElement,MarkovChain,Distribution)
end
#    InterestRate::R,
#    LaborSupply = 1.0,
#    β = 0.96,
#    d = 0.025,
#    A = 1.00,
#    B_ = 0.0,
#    γc = 1.00,
#    γl = 1.0,
#    α = 0.36,
#    T = 1.0,
#    ζ = 10000000000.0,
#    GridSize = 40,
#    GridMax = 50,
#    IndividualStates = [1.0;0.2],
#    NumberOfQuadratureNodesPerElement = 2

MarkovElement =  AyiagariModel(0.01)
na,ns = MarkovElement.FiniteElement.na,MarkovElement.MarkovChain.ns
ResidualSize = na*ns
#NumberOfHouseholds = 400

pol = SolveFiniteElement(0.01,MarkovElement.Guess,MarkovElement)
#=
req,θeq,capeq,sr = equilibrium(MarkovElement)

###Plot the Distribution
res,EA,pdf1 = StationaryDistribution(req,θeq,MarkovElement)
DistGrid = MarkovElement.Distribution.DistributionAssetGrid
DistSize = MarkovElement.Distribution.DistributionSize
p1 = plot(DistGrid,pdf1[1:DistSize],label="employed")
p1 = plot!(DistGrid,pdf1[DistSize+1:2*DistSize],label="umemployed")
p1 = plot!(xlims=(0.0,30.0))
#savefig(p,"PlotDistribution2.pdf")


################ Plot solution
p2 = plot(MarkovElement.FiniteElement.nodes,θeq[1:na], label="employed")
p2 = plot!(MarkovElement.FiniteElement.nodes,θeq[na+1:2*na],label="unemeployed")
p2 = plot!(MarkovElement.FiniteElement.nodes,MarkovElement.FiniteElement.nodes,label="45 degree line", line = :dot)
p2 = plot!(legend=:topleft)
#savefig(p,"PlotSolution2.pdf")


#dem,sup,ir = PlotEquilibrium(MarkovElement)
#p3 = plot(dem,ir,label= "Demand")
#p3 = plot!(sup,ir,label = "Supply")
#p = plot(p1,p2,p3, layout=(1,3),size=(1000,400))
#savefig(p,"exosolution.pdf")


############################ Some tests 
MarkovElement =  AyiagariModel(0.01,1.0,0.96,0.025,1.00,0.0,1.0,1.0,0.36,1.0,10.0,40,50,[1.0;0.2],2)
θstar = SolveFiniteElement(0.025,MarkovElement.Guess,MarkovElement)
#res,EA,pdf1 = StationaryDistribution(0.025,θstar,MarkovElement)
#println(θstar)
#println(EA)
polm = reshape(θstar,na,ns)
DistGrid = MarkovElement.Distribution.DistributionAssetGrid
DistSize = MarkovElement.Distribution.DistributionSize
#p1 = plot(DistGrid,pdf1[:,1],label="umemployed")
#p2 = plot(DistGrid,pdf1[:,2],label="employed")
#p = plot(p1,p2, layout=(2,1),legend = true)
#savefig(p,"PlotDistribution2.pdf")


#@show WeightedResidual(MarkovElement.Guess,0.005,MarkovElement)[1][1:MarkovElement.FiniteElement.na]
#@show WeightedResidual(MarkovElement.Guess,0.005,MarkovElement)[1][MarkovElement.FiniteElement.na+1:2*MarkovElement.FiniteElement.na]



#println(sr)

#println(EA)

###Plot guess
#=
fin = 8
p = plot(MarkovElement.FiniteElement.nodes[1:fin],MarkovElement.GuessMatrix[1:fin,1], label = "employed")
p = plot!(MarkovElement.FiniteElement.nodes[1:fin],MarkovElement.GuessMatrix[1:fin,2], label = "unemployed")
p = plot!(MarkovElement.FiniteElement.nodes[1:fin],MarkovElement.FiniteElement.nodes[1:fin], line = :dot)
savefig(p,"PlotGuess2.pdf")
=#

#=
############################# Checking on the Jacobian 
resj =  WeightedResidual(MarkovElement.Guess,0.025,MarkovElement)[2]
@show Calculus.gradient(x -> WeightedResidual(x,0.025,MarkovElement)[1][1],MarkovElement.Guess)
jacobian = zleros(ResidualSize,ResidualSize)
for i = 1:ResidualSize
    jacobian[i,:] = Calculus.gradient(x -> WeightedResidual(x,0.025,MarkovElement)[1][i],MarkovElement.Guess)
end
@show LinearAlgebra.norm(jacobian - resj,2)
@show jacobian[11:15,:] - resj[11:15,:] =#

