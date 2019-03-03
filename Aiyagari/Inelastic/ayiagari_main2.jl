#using QuantEcon
using Plots
using FastGaussQuadrature
using BenchmarkTools
using ForwardDiff
using Calculus
using DelimitedFiles
#Include ayiagari functions and types
include("ayiagari2.jl")

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
            assets = (1.0 + r)*nodes[i] + w*IndividualStates[j]  - B_ - 0.1  
            assets > B_ ? Guess[n] = 24.0/25.0 * assets : Guess[n] = B_
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


MarkovElement =  AyiagariModel(0.01)
na,ns = MarkovElement.FiniteElement.na,MarkovElement.MarkovChain.ns
ResidualSize = na*ns
#NumberOfHouseholds = 400


############################ Some tests 
#= 
θstar = SolveFiniteElement(0.025,MarkovElement.Guess,MarkovElement)
res,EA,pdf1 = StationaryDistribution(0.025,θstar,MarkovElement)
println(θstar)
println(EA)
DistGrid = MarkovElement.Distribution.DistributionAssetGrid
DistSize = MarkovElement.Distribution.DistributionSize
p1 = plot(DistGrid,pdf1[:,1],label="umemployed")
p2 = plot(DistGrid,pdf1[:,2],label="employed")
p = plot(p1,p2, layout=(2,1),legend = true)
savefig(p,"PlotDistribution2.pdf")
=#


#@show WeightedResidual(MarkovElement.Guess,0.005,MarkovElement)[1][1:MarkovElement.FiniteElement.na]
#@show WeightedResidual(MarkovElement.Guess,0.005,MarkovElement)[1][MarkovElement.FiniteElement.na+1:2*MarkovElement.FiniteElement.na]


req,θeq,capeq,sr = equilibrium(MarkovElement)
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


dem,sup,ir = PlotEquilibrium(MarkovElement)
p3 = plot(dem,ir,label= "Demand")
p3 = plot!(sup,ir,label = "Supply")
p = plot(p1,p2,p3, layout=(1,3),size=(1000,400))
savefig(p,"exosolution.pdf")