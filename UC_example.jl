# Define the packages
using JuMP # used for mathematical programming
using Gurobi # used for solving

# Define some input data about the test system
# Maximum power output of generators
P_max = [1000,1000];
# Minimum power output of generators
P_min = [500,100];
# Marginal cost of generators
MC = [30,50];
# Total demand
D = [1600,1000];
# Wind forecast
WF = [200,800];


# Function solve_uc() solves the unit commitment problem
function solve_uc(P_max, P_min, MC, D, WF)
    #Define the unit commitment (UC) model
    uc = Model(optimizer_with_attributes(Gurobi.Optimizer,"OutputFlag" =>1))

    # Define decision variables
    @variable(uc, g[i=1:2,t=1:2] >= 0) # power output of generators
    @variable(uc, z[i=1:2,t=1:2], Bin) # Binary status of generators
    @variable(uc, w[t=1:2] >= 0 ) # wind power injection

    # Define the objective function
    @objective(uc,Min,sum(sum(MC[i] * g[i,t] for i=1:2) for t=1:2))

    # Define the constraint on the maximum and minimum power output of each generator
    for i in 1:2
        @constraint(uc, [i=1:2,t=1:2], g[i,t] <= P_max[i] * z[i,t]) #maximum
        @constraint(uc, [i=1:2,t=1:2], g[i,t] >= P_min[i] * z[i,t]) #minimum
    end

    # Define the constraint on the wind power injection
    @constraint(uc, [t=1:2], w[t] <= WF[t])

    # Define the power balance constraint
    @constraint(uc, [t=1:2], sum(g[i,t] for i=1:2) + w[t] == D[t])

    # Solve statement
    status = optimize!(uc)

    return status, value.(g), value.(w), value.(z), objective_value(uc)
end

# Solve the economic dispatch problem
status,g_opt,w_opt,z_opt,obj=solve_uc(P_max, P_min, MC, D, WF);


println("\n")
println("Dispatch of Generators: ", transpose(g_opt), " MW")
println("Commitments of Generators: ", transpose(z_opt))
println("Dispatch of Wind: ", transpose(w_opt), " MW")
println("\n")
println("Total cost: ", obj, "\$")
