## Simplified Generation Expansion Planning with Energy Storage Systems (Greenfield & Brownfield)
# author: Olivier Deforche & Bas Carpentero
# last update: December 2022

## Instructions on how to run the code
# 1) Packages: Normally, the packages necessary should be installed automatically via the Manifest.toml file.
# However, if necessary, it is possible to add packages by pressing: "alt gr + ]", after which you can enter "add [package name]"
# 2) Solvers: There are three solvers out of which the runner can choose: Gurobi, Ipopt and Clp. Note that Gurobi is not free, 
# and therefore it needs additional installation with licensing. (https://www.gurobi.com/solutions/gurobi-optimizer/)
# 3) Type: The model is able to be implemented as a Greenfield or Brownfield, this can be selected at the build step.
# 4) Output: The output are several graphs split up into three parts: 1 - outpout of the model, 2 - Lagrangian parameters (equality)
# 3 - Lagrangian parameters (inequalty). The output gets saved in the 'current' directory julia is active in.   

## Step 0: Activate environment - ensure consistency accross computers
using Pkg
Pkg.activate(@__DIR__) # @__DIR__ = directory this script is in
Pkg.instantiate() # If a Manifest.toml file exist in the current project, download all the packages declared in that manifest. Else, resolve a set of feasible packages from the Project.toml files and install them.

##  Step 1: input data
using CSV
using DataFrames
using YAML

data = YAML.load_file(joinpath(@__DIR__, "data_gep_ESS.yaml"))
repr_days = CSV.read(joinpath(@__DIR__, "Weights_12_reprdays.csv"), DataFrame)
ts = CSV.read(joinpath(@__DIR__, "Profiles_12_reprdays.csv"), DataFrame)

## Step 2: create model & pass data to model
using JuMP
using Gurobi
using Ipopt
using Clp

# Choose Optimizer
# m = Model(optimizer_with_attributes(Gurobi.Optimizer))
# m = Model(optimizer_with_attributes(Ipopt.Optimizer))
m = Model(optimizer_with_attributes(Clp.Optimizer))
# JuMP.set_optimizer_attribute(m, "LogLevel", 4)
# JuMP.set_optimizer_attribute(m, "Algorithm", 4)

# Step 2a: create sets
function define_sets!(m::Model, data::Dict)
    # create dictionary to store sets
    m.ext[:sets] = Dict()

    # define the sets
    m.ext[:sets][:JH] = 1:data["nTimesteps"] # Timesteps
    m.ext[:sets][:JD] = 1:data["nReprDays"] # Representative days
    m.ext[:sets][:ID] = [id for id in keys(data["dispatchableGenerators"])] # dispatchable generators
    m.ext[:sets][:IV] = [iv for iv in keys(data["variableGenerators"])] # variable generators
    m.ext[:sets][:I] = union(m.ext[:sets][:ID], m.ext[:sets][:IV]) # all generators
    m.ext[:sets][:Z] = [z for z in keys(data["ESS"])] # variable generators #Change

    # return model
    return m
end

# Step 2b: add time series
function process_time_series_data!(m::Model, data::Dict, ts::DataFrame)
    # extract the relevant sets
    IV = m.ext[:sets][:IV] # Variable generators
    JH = m.ext[:sets][:JH] # Time steps
    JD = m.ext[:sets][:JD] # Days

    # create dictionary to store time series
    m.ext[:timeseries] = Dict()
    m.ext[:timeseries][:AF] = Dict()

    # example: add time series to dictionary
    m.ext[:timeseries][:D] = [ts.Load[jh+data["nTimesteps"]*(jd-1)] for jh in JH, jd in JD]
    m.ext[:timeseries][:AF][IV[1]] = [ts.LFW[jh+data["nTimesteps"]*(jd-1)] for jh in JH, jd in JD]
    m.ext[:timeseries][:AF][IV[2]] = [ts.LFS[jh+data["nTimesteps"]*(jd-1)] for jh in JH, jd in JD] 

    # return model
    return m
end

# step 2c: process input parameters
function process_parameters!(m::Model, data::Dict, repr_days::DataFrame)
    # extract the sets you need
    I = m.ext[:sets][:I]
    ID = m.ext[:sets][:ID]
    IV = m.ext[:sets][:IV]
    Z = m.ext[:sets][:Z] #Change

    # generate a dictonary "parameters"
    m.ext[:parameters] = Dict()

    # input parameters
    ??CO2 = m.ext[:parameters][:??CO2] = data["CO2Price"] #euro/ton
    m.ext[:parameters][:VOLL] = data["VOLL"] #VOLL
    r = m.ext[:parameters][:discountrate] = data["discountrate"] #discountrate
    m.ext[:parameters][:W] = repr_days.Weights # wieghts of each representative date

   
    d = merge(data["dispatchableGenerators"],data["variableGenerators"])
    ?? = m.ext[:parameters][:??] = Dict(i => d[i]["fuelCosts"] for i in ID) #EUR/MWh
    ?? = m.ext[:parameters][:??] = Dict(i => d[i]["emissions"] for i in ID) #ton/MWh
    m.ext[:parameters][:VC] = Dict(i => ??[i]+??CO2*??[i] for i in ID) # variable costs - EUR/MWh

    
    ESS = data["ESS"]
    m.ext[:parameters][:??_sup] = Dict(z => ESS[z]["eff_sup"] for z in Z) # ESS supply efficiencies #Change 
    m.ext[:parameters][:??_abs] = Dict(z => ESS[z]["eff_abs"] for z in Z) # ESS absorb efficiencies #Change 
    m.ext[:parameters][:E_max] = Dict(z => ESS[z]["stored_energy_max"] for z in Z) # maximal stored capacity for respective ESS #Change 
    m.ext[:parameters][:SOC_min] = Dict(z => ESS[z]["SOC_min"] for z in Z) # minimum state of charge of ESS type s unit j (%) #Change 
    m.ext[:parameters][:SOC_max] = Dict(z => ESS[z]["SOC_max"] for z in Z) # maximum state of charge of ESS type s unit j (%) #Change 
    m.ext[:parameters][:p_abs_min] = Dict(z => ESS[z]["p_abs_min"] for z in Z) # minimal internal power input #Change 
    m.ext[:parameters][:p_abs_max] = Dict(z => ESS[z]["p_abs_max"] for z in Z) # maximal internal power input #Change 
    m.ext[:parameters][:p_sup_min] = Dict(z => ESS[z]["p_sup_min"] for z in Z) # minimal internal power output #Change 
    m.ext[:parameters][:p_sup_max] = Dict(z => ESS[z]["p_sup_max"] for z in Z) # maximal internal power output #Change 
    
    # Investment costs
    OC = m.ext[:parameters][:OC] = Dict(i => d[i]["OC"] for i in I) # EUR/MW
    LifeTime = m.ext[:parameters][:LT] = Dict(i => d[i]["lifetime"] for i in I) # years
    m.ext[:parameters][:IC] = Dict(i => r*OC[i]/(1-(1+r).^(-LifeTime[i])) for i in I) # EUR/MW/y

    # legacy capacity
    LC = m.ext[:parameters][:LC] = Dict(i => d[i]["legcap"] for i in I) # MW

    # return model
    return m
end

# call functions
define_sets!(m, data)
process_time_series_data!(m, data, ts)
process_parameters!(m, data, repr_days)


## Step 3: construct your model
# Greenfield GEP - single year based on representative days
function build_greenfield_1Y_GEP_model!(m::Model)
    # Clear m.ext entries "variables", "expressions" and "constraints"
    m.ext[:variables] = Dict()
    m.ext[:expressions] = Dict()
    m.ext[:constraints] = Dict()

    # Extract sets
    I = m.ext[:sets][:I]
    ID = m.ext[:sets][:ID]
    IV = m.ext[:sets][:IV]
    JH = m.ext[:sets][:JH]
    JD = m.ext[:sets][:JD]
    Z = m.ext[:sets][:Z] #Change

    # Extract time series data
    D = m.ext[:timeseries][:D] # demand
    AF = m.ext[:timeseries][:AF] # availabilim.ty factors

    # Extract parameters
    VOLL = m.ext[:parameters][:VOLL] # VOLL
    VC = m.ext[:parameters][:VC] # variable cost
    IC = m.ext[:parameters][:IC] # investment cost
    W = m.ext[:parameters][:W] # weights
    ??_sup = m.ext[:parameters][:??_sup] # ESS supply efficiencies #Change 
    ??_abs = m.ext[:parameters][:??_abs] # ESS absorb efficiencies #Change 
    E_max = m.ext[:parameters][:E_max] # maximal stored capacity for respective ESS #Change 
    SOC_min = m.ext[:parameters][:SOC_min] # minimum state of charge of ESS type s unit j (%) #Change 
    SOC_max = m.ext[:parameters][:SOC_max] # maximum state of charge of ESS type s unit j (%) #Change 
    P_abs_min = m.ext[:parameters][:p_abs_min] # minimal internal power input #Change 
    P_abs_max = m.ext[:parameters][:p_abs_max] # maximal internal power input #Change 
    P_sup_min = m.ext[:parameters][:p_sup_min] # minimal internal power output #Change 
    P_sup_max = m.ext[:parameters][:p_sup_max] # maximal internal power output #Change 

    # Create variablesabs
    cap = m.ext[:variables][:cap] = @variable(m, [i=I], lower_bound=0, base_name="capacity")
    g = m.ext[:variables][:g] = @variable(m, [i=I,jh=JH,jd=JD], lower_bound=0, base_name="generation")
    ens = m.ext[:variables][:ens] = @variable(m, [jh=JH,jd=JD], lower_bound=0, base_name="energy_not_served")
    P_abs = m.ext[:variables][:P_abs] = @variable(m, [z=Z,jh=JH,jd=JD], lower_bound=0, base_name="power_asborbed") #Change
    P_sup = m.ext[:variables][:P_sup] = @variable(m, [z=Z,jh=JH,jd=JD], lower_bound=0, base_name="power_supplied") #Change 
    E = m.ext[:variables][:E] = @variable(m, [z=Z,jh=JH,jd=JD], lower_bound=0, base_name="power_stored") #Change

    # Create affine expressions (= linear combinations of variables)
    curt = m.ext[:expressions][:curt] = @expression(m, [i=IV,jh=JH,jd=JD],
        AF[i][jh,jd]*cap[i] - g[i,jh,jd]
    )
    
    # Formulate objective 1a
    m.ext[:objective] = @objective(m, Min,
        + sum(IC[i]*cap[i] for i in I)
        + sum(W[jd]*VC[i]*g[i,jh,jd] for i in ID, jh in JH, jd in JD)
        + sum(W[jd]*ens[jh,jd]*VOLL for jh in JH, jd in JD)
    )

    # 2a - power balance
    m.ext[:constraints][:con2a] = @constraint(m, [jh=JH,jd=JD],
        sum(g[i,jh,jd] for i in I) + sum(P_sup[z,jh,jd]*??_sup[z] for z in Z) == 
        D[jh,jd] - ens[jh,jd] + sum(P_abs[z,jh,jd]/??_abs[z] for z in Z)
    )
    
    # 3a1 - renewables 
    m.ext[:constraints][:con3a1res] = @constraint(m, [i=IV,jh=JH,jd=JD],
    g[i,jh,jd] <= AF[i][jh,jd]*cap[i]
    )

    # 3a1 - conventional
    m.ext[:constraints][:con3a1conv] = @constraint(m, [i=ID,jh=JH,jd=JD],
        g[i,jh,jd] <= cap[i]
    )

    #4a - ESS balance constraint #Change
    m.ext[:constraints][:con4a] = @constraint(m, [z=Z,jh=JH[2:end],jd=JD],
        E[z,jh,jd] == E[z,jh-1,jd] + P_abs[z,jh,jd] - P_sup[z,jh,jd]
    )

    #4b1 - 1st hour of the day constraint (Paper) #Change
    m.ext[:constraints][:con4b1] = @constraint(m, [z=Z,jd=JD],
        E[z,1,jd] == E_max[z]*SOC_min[z]
    )

    #4b2 - 24th hour of the day constraint (Paper) #Change
    m.ext[:constraints][:con4b2] = @constraint(m, [z=Z,jd=JD],
        E[z,24,jd] == E_max[z]*SOC_min[z]
    ) 

    #4b3 - 1st hour of the day constraint energy supply #Change
    m.ext[:constraints][:con4b3] = @constraint(m, [z=Z,jd=JD],
        P_sup[z,1,jd] == 0
    )
    
    #4c1 - Power absorbed #Change
    m.ext[:constraints][:con4c1] = @constraint(m, [z=Z,jh=JH,jd=JD],
        P_abs_min[z] <= P_abs[z,jh,jd] <= P_abs_max[z]  
    )

    #4c2 - Power supplied #Change
    m.ext[:constraints][:con4c2] = @constraint(m, [z=Z,jh=JH,jd=JD],
        P_sup_min[z] <= P_sup[z,jh,jd] <= P_sup_max[z]  
    )

    #4c3 - State of charge #Change
    m.ext[:constraints][:con4c3] = @constraint(m, [z=Z,jh=JH,jd=JD],
        E_max[z]*SOC_min[z] <= E[z,jh,jd] <= E_max[z]*SOC_max[z] 
    )

    return m
end


# Brownfield GEP - single year based on representative days
function build_brownfield_1Y_GEP_model!(m::Model)

    # start from Greenfield
    m = build_greenfield_1Y_GEP_model!(m::Model)

    # extract sets
    ID = m.ext[:sets][:ID]
    IV = m.ext[:sets][:IV]
    JH = m.ext[:sets][:JH]
    JD = m.ext[:sets][:JD]  
    
    # Extract parameters
    LC = m.ext[:parameters][:LC]

    # Extract time series
    AF = m.ext[:timeseries][:AF] # availability factors

    # extract variables
    g = m.ext[:variables][:g]
    cap = m.ext[:variables][:cap]

    # remove the constraints that need to be changed:
    for iv in IV, jh in JH, jd in JD
        delete(m,m.ext[:constraints][:con3a1res][iv,jh,jd])
    end
    for id in ID, jh in JH, jd in JD
        delete(m,m.ext[:constraints][:con3a1conv][id,jh,jd])
    end

    # define new constraints
    # 3a1 - renewables
    m.ext[:constraints][:con3a1res] = @constraint(m, [i=IV, jh=JH, jd =JD],
        g[i,jh,jd] <= AF[i][jh,jd]*(cap[i]+LC[i])
    )

    # 3a1 - conventional
    m.ext[:constraints][:con3a1conv] = @constraint(m, [i=ID, jh=JH, jd=JD],
        g[i,jh,jd] <= (cap[i]+LC[i])
    )
    return m
end

# Build your model
build_greenfield_1Y_GEP_model!(m)
# build_brownfield_1Y_GEP_model!(m)


## Step 4: solver
optimize!(m)


# check termination status
print(
    """

    Termination status: $(termination_status(m))

    """
)


# print some output
@show value(m.ext[:objective])
@show value.(m.ext[:variables][:cap])

## Step 5: interpretation
using Plots
using Interact
using StatsPlots

# examples on how to access data

# sets
JH = m.ext[:sets][:JH]
JD = m.ext[:sets][:JD]
I = m.ext[:sets][:I]
IV = m.ext[:sets][:IV]
ID = m.ext[:sets][:ID]
Z = m.ext[:sets][:Z] #Change

# parameters
D = m.ext[:timeseries][:D]
AF = m.ext[:timeseries][:AF] 
Wind = AF["Wind"]
Solar = AF["Solar"]
W = m.ext[:parameters][:W]
LC = m.ext[:parameters][:LC]
# variables/expressions
cap = value.(m.ext[:variables][:cap])
g = value.(m.ext[:variables][:g])
ens = value.(m.ext[:variables][:ens])
curt = value.(m.ext[:expressions][:curt])
?? = dual.(m.ext[:constraints][:con2a])
P_abs =value.(m.ext[:variables][:P_abs]) #Change
P_sup = value.(m.ext[:variables][:P_sup]) #Change
E = value.(m.ext[:variables][:E]) #Change

# create arrays for plotting
??vec = [??[jh,jd]/W[jd] for jh in JH, jd in JD]
gvec = [g[i,jh,jd] for i in I, jh in JH, jd in JD]
capvec = [cap[i] for  i in I]
ESSvec = [E[z,jh,jd] for z in Z, jh in JH, jd in JD] #Change
P_absvec  = [P_abs[z,jh,jd] for z in Z, jh in JH, jd in JD] #Change
P_supvec = [P_sup[z,jh,jd] for z in Z, jh in JH, jd in JD] #Change



for jd in 1:12

    # Electricity price 
    p1 = plot(JH,??vec[:,jd], xlabel="Timesteps [-]", ylabel="?? [EUR/MWh]", label="?? [EUR/MWh]", legend=:outertopright);

    # Dispatch
    p2 = groupedbar(transpose(gvec[:,:,jd]), label=["Mid" "Base" "Peak" "Wind" "Solar"], bar_position = :stack,legend=:outertopright,ylims=(0,13_000));
    plot!(p2, JH, D[:,jd], label ="Demand", xlabel="Timesteps [-]", ylabel="Generation [MWh]", legend=:outertopright, lindewidth=3, lc=:black);

    # Capacity
    p3 = bar(capvec, label="", xticks=(1:length(capvec), ["Mid" "Base" "Peak" "Wind" "Solar"]), xlabel="Technology [-]", ylabel="New capacity [MW]", legend=:outertopright);

    # ESS
    p4 = groupedbar(transpose(ESSvec[:,:,jd]), label=["BESS    " "PSH"], xlabel="Timesteps [-]", ylabel="Storage [MWh]", bar_position = :stack,legend=:outertopright,ylims=(0,8_000));
    plot!(p4, JH, P_absvec[1,:,jd]+P_absvec[2,:,jd], label ="P abs", legend=:outertopright, lindewidth=3, lc=:black);
    plot!(p4, JH, P_supvec[1,:,jd]+P_supvec[2,:,jd], label ="P sup", legend=:outertopright, lindewidth=3, lc=:red);

    # p4 = groupedbar(transpose(ESSvec[:,:,jd]), label=["PSH"], bar_position = :stack,legend=:outertopright,ylims=(0,13_000));

    # AF
    p5 = plot(JH, Wind[:,jd],  label ="Wind", lindewidth=3, lc=:black);
    plot!(p5, JH, Solar[:,jd], label ="Solar", xlabel="Timesteps [-]", ylabel="AF [0-1]", legend=:outertopright, lindewidth=3, lc=:red);



    # combine
    plot(p1, p2, p3, p4,p5, layout = (5,1))
    # plot!()
    plot!(size=(700,700))

    # Save plots
    # savefig("day"*string(jd)*"Gurobi")
    # savefig("day"*string(jd)*"Ipopt")
    savefig("day"*string(jd)*"Clp")
    # savefig("day"*string(jd)*"Gurobi"*"Brown")
    # savefig("day"*string(jd)*"ipopt"*"Brown")
    # savefig("day"*string(jd)*"Clp"*"Brown")

end
print("Output done")

# Lagrangian parameters equality constraints
??2 = dual.(m.ext[:constraints][:con4a])
??3 = dual.(m.ext[:constraints][:con4b1])
??4 = dual.(m.ext[:constraints][:con4b2])
??5 = dual.(m.ext[:constraints][:con4b3])

zero = zeros(1,12)
??2vec_PSH = [??2["PSH",:,:]; zero]
??2vec_PSH = [??2vec_PSH[jh,jd]/W[jd] for jh in JH, jd in JD]
??2vec_BESS = [??2["BESS",:,:]; zero]
??2vec_BESS = [??2vec_BESS[jh,jd]/W[jd] for jh in JH, jd in JD]
??3vec_PSH = [??3["PSH",jd]/W[jd] for jd in JD]
??3vec_BESS = [??3["BESS",jd]/W[jd] for jd in JD]
??4vec_PSH = [??4["PSH",jd]/W[jd] for jd in JD]
??4vec_BESS = [??4["BESS",jd]/W[jd] for jd in JD]
??5vec_PSH = [??5["PSH",jd]/W[jd] for jd in JD]
??5vec_BESS = [??5["BESS",jd]/W[jd] for jd in JD]


for jd in 1:12
    p1 = plot(JH,??2vec_PSH[:,jd], label="??2 PSH [?]", xlabel="Timesteps [-]", ylabel="??2 PSH [M]", legend=:outertopright, lindewidth=3, lc=:red);
    p2 = plot(JH, ??2vec_BESS[:,jd], label ="??2 BESS", xlabel="Timesteps [-]", ylabel="??2 BESS [M]", legend=:outertopright, lindewidth=3, lc=:red);
    # combine
    plot(p1, p2, layout = (2,1))
    # plot!()
    plot!(size=(700,700))

    # Save plots
    # savefig("day"*string(jd)*"Gurobi"*"lagrange"*"eq")
    # savefig("day"*string(jd)*"Ipopt"*"lagrange"*"eq")
    savefig("day"*string(jd)*"Clp"*"lagrange"*"eq")
    # savefig("day"*string(jd)*"Gurobi"*"Brown"*"lagrange"*"eq")
    # savefig("day"*string(jd)*"ipopt"*"Brown"*"lagrange"*"eq")
    # savefig("day"*string(jd)*"Clp"*"Brown"*"lagrange"*"eq")

end
print("Equalities done")

p1 = plot(JD, ??3vec_PSH[:], label ="??3 PSH", xlabel="Timesteps [-]", ylabel="??3 PSH [M]", legend=:outertopright, lindewidth=3, lc=:red);
p2 = plot(JD, ??3vec_BESS[:], label ="??3 BESS", xlabel="Timesteps [-]", ylabel="??3 BESS [M]", legend=:outertopright, lindewidth=3, lc=:red);
p3= plot(JD, ??4vec_PSH[:], label ="??4 PSH", xlabel="Timesteps [-]", ylabel="??4 PSH [M]", legend=:outertopright, lindewidth=3, lc=:red);
p4 = plot(JD, ??4vec_BESS[:], label ="??4 BESS", xlabel="Timesteps [-]", ylabel="??4 BESS [M]", legend=:outertopright, lindewidth=3, lc=:red);
p5 = plot(JD, ??5vec_PSH[:], label ="??5 PSH", xlabel="Timesteps [-]", ylabel="??5 PSH [M]", legend=:outertopright, lindewidth=3, lc=:red);
p6 = plot(JD, ??5vec_BESS[:], label ="??5 BESS", xlabel="Timesteps [-]", ylabel="??5 BESS [M]", legend=:outertopright, lindewidth=3, lc=:red);

plot(p1, p2, p3, p4,p5,p6, layout = (6,1))
plot!(size=(700,700))

# savefig("Gurobi"*"lagrange"*"eq")
# savefig("Ipopt"*"lagrange"*"eq")
savefig("Clp"*"lagrange"*"eq")
# savefig("Gurobi"*"Brown"*"lagrange"*"eq")
# savefig("ipopt"*"Brown"*"lagrange"*"eq")
# savefig("Clp"*"Brown"*"lagrange"*"eq")


# Lagrangian parameters inequality constraints
??4 = dual.(m.ext[:constraints][:con3a1res])
??5 = dual.(m.ext[:constraints][:con3a1conv])
??6 = dual.(m.ext[:constraints][:con4c1])
??7 = dual.(m.ext[:constraints][:con4c1])
??8 = dual.(m.ext[:constraints][:con4c2])
??9 = dual.(m.ext[:constraints][:con4c2])
??10 = dual.(m.ext[:constraints][:con4c3])
??11 = dual.(m.ext[:constraints][:con4c3])



??4vec = [??4[i,jh,jd]/W[jd] for i in IV, jh in JH, jd in JD]
??5vec = [??5[i,jh,jd]/W[jd] for i in ID, jh in JH, jd in JD]
??6vec = [??6[z,jh,jd]/W[jd] for z in Z, jh in JH, jd in JD]
??8vec = [??8[z,jh,jd]/W[jd] for z in Z, jh in JH, jd in JD]
??10vec = [??10[z,jh,jd]/W[jd] for z in Z, jh in JH, jd in JD]



for jd in 1:12
    p1 = plot(JH, ??4vec[1,:,jd], label ="Wind", xlabel="Timesteps [-]", ylabel="??4", legend=:outertopright, lindewidth=3, lc=:red);
    plot!(p1, JH,  ??4vec[2,:,jd], label ="Solar", xlabel="Timesteps [-]", ylabel="??4", legend=:outertopright, lindewidth=3, lc=:black);
    p2 = plot(JH, ??5vec[1,:,jd], label ="Mid", xlabel="Timesteps [-]", ylabel="??5", legend=:outertopright, lindewidth=3, lc=:red);
    plot!(p2, JH,  ??5vec[2,:,jd], label ="Base", xlabel="Timesteps [-]", ylabel="??5", legend=:outertopright, lindewidth=3, lc=:black);
    plot!(p2, JH,  ??5vec[3,:,jd], label ="Peak", xlabel="Timesteps [-]", ylabel="??5", legend=:outertopright, lindewidth=3, lc=:yellow);
    p3 = plot(JH, ??6vec[1,:,jd], label ="BESS", xlabel="Timesteps [-]", ylabel="??6", legend=:outertopright, lindewidth=3, lc=:red);
    plot!(p3, JH,  ??6vec[2,:,jd], label ="PSH", xlabel="Timesteps [-]", ylabel="??6", legend=:outertopright, lindewidth=3, lc=:black);
    p4 = plot(JH, ??8vec[1,:,jd], label ="BESS", xlabel="Timesteps [-]", ylabel="??8", legend=:outertopright, lindewidth=3, lc=:red);
    plot!(p4, JH,  ??8vec[2,:,jd], label ="PSH", xlabel="Timesteps [-]", ylabel="??8", legend=:outertopright, lindewidth=3, lc=:black);
    p5 = plot(JH, ??10vec[1,:,jd], label ="BESS", xlabel="Timesteps [-]", ylabel="??10", legend=:outertopright, lindewidth=3, lc=:red);
    plot!(p5, JH,  ??10vec[2,:,jd], label ="PSH", xlabel="Timesteps [-]", ylabel="??10", legend=:outertopright, lindewidth=3, lc=:black);
    
    # combine
    plot(p1, p2,p3,p4,p5, layout = (5,1))
    # plot!()
    plot!(size=(700,700))

    # Save plots
    # savefig("day"*string(jd)*"Gurobi"*"lagrange"*"ineq")
    # savefig("day"*string(jd)*"Ipopt"*"lagrange""ineq")
    savefig("day"*string(jd)*"Clp"*"lagrange"*"ineq")
    # savefig("day"*string(jd)*"Gurobi"*"Brown"*"lagrange""ineq")
    # savefig("day"*string(jd)*"ipopt"*"Brown"*"lagrange""ineq")
    # savefig("day"*string(jd)*"Clp"*"Brown"*"lagrange""ineq")

end
print("Inequalities done")
