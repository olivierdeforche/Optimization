## Backbone to structure your code
# author: Kenneth Bruninx
# last update: October 26, 2020
# description: backbone to structure your code. You're not obliged to use this
# in your assignment, but you may.

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

print(repr_days)
print(ts)

## Step 2: create model & pass data to model
using JuMP
using Gurobi
m = Model(optimizer_with_attributes(Gurobi.Optimizer))

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
    αCO2 = m.ext[:parameters][:αCO2] = data["CO2Price"] #euro/ton
    m.ext[:parameters][:VOLL] = data["VOLL"] #VOLL
    r = m.ext[:parameters][:discountrate] = data["discountrate"] #discountrate
    m.ext[:parameters][:W] = repr_days.Weights # wieghts of each representative date

   
    d = merge(data["dispatchableGenerators"],data["variableGenerators"])
    β = m.ext[:parameters][:β] = Dict(i => d[i]["fuelCosts"] for i in ID) #EUR/MWh
    δ = m.ext[:parameters][:δ] = Dict(i => d[i]["emissions"] for i in ID) #ton/MWh
    m.ext[:parameters][:VC] = Dict(i => β[i]+αCO2*δ[i] for i in ID) # variable costs - EUR/MWh

    
    ESS = data["ESS"]
    m.ext[:parameters][η_sup] = Dict(z => ESS[z]["eff_sup"] for z in Z) # ESS supply efficiencies #Change 
    m.ext[:parameters][η_abs] = Dict(z => ESS[z]["eff_abs"] for z in Z) # ESS absorb efficiencies #Change 
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
# Greenfield GEP - single year (Lecture 3 - slide 25, but based on representative days instead of full year)
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
    η_sup = m.ext[:parameters][η_sup] # ESS supply efficiencies #Change 
    η_abs = m.ext[:parameters][η_abs] # ESS absorb efficiencies #Change 
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
        sum(g[i,jh,jd] for i in I) + sum(P_sup[z,jh,jd]*η_sup[z] for z in Z) == D[jh,jd] - ens[jh,jd] + sum(P_abs[z,jh,jd]/η_abs[z] for z in Z)#Change
    )

    # 2c2 - load shedding
    m.ext[:constraints][:con2c] = @constraint(m, [jh=JH,jd=JD],
        ens[jh,jd] <= D[jh,jd]
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
    m.ext[:constraints][:con4a] = @constraint(m, [z=Z,jh=JH[2:],jd=JD],
        E[z,jh,jd] = E[z,jh-1,jd] + P_abs[z,jh,jd] - P_sup[z,jh,jd]
    )

    #4b1 - 1st hour of the day constraint #Change
    m.ext[:constraints][:con4b1] = @constraint(m, [z=Z,jd=JD],
        E[z,1,jd] = E[z,24,jd]
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
    m.ext[:constraints][:con4b2] = @constraint(m, [z=Z,jh=JH,jd=JD],
        E_max[z]*SOC_min[z] <= E[z,jh,jd] <= E_max[z]*SOC_max[z] 
    )

    return m
end


# Brownfield GEP - single year
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
        g(i,jh,jd) <= AF[i][jh,jd]*(cap[i]+LC[i])
    )

    # 3a1 - conventional
    m.ext[:constraints][:con3a1conv] = @constraint(m, [i=ID, jh=JH, jd=JD],
        g[i,jh,jd] <= (cap[i]+LC[i])
    )

    return m
end

# Build your model
# build_greenfield_1Y_GEP_model!(m)
build_brownfield_1Y_GEP_model!(m)

## Step 4: solve
# current model is incomplete, so all variables and objective will be zero
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
Z = m.ext[:sets][:Z] #Change

# parameters
D = m.ext[:timeseries][:D]
W = m.ext[:parameters][:W]
LC = m.ext[:parameters][:LC]

# variables/expressions
cap = value.(m.ext[:variables][:cap])
g = value.(m.ext[:variables][:g])
ens = value.(m.ext[:variables][:ens])
curt = value.(m.ext[:expressions][:curt])
λ = dual.(m.ext[:constraints][:con2a])
P_abs =value.(m.ext[:variables][:P_abs]) #Change
P_sup = value.(m.ext[:variables][:P_sup]) #Change
E = value.(m.ext[:variables][:E]) #Change

# create arrays for plotting
λvec = [λ[jh,jd]/W[jd] for jh in JH, jd in JD]
gvec = [g[i,jh,jd] for i in I, jh in JH, jd in JD]
capvec = [cap[i] for  i in I]
ESSvec = [E[z,jh,jd] for z in Z, jh in JH, jd in JD] #Change
P_absvec  = [P_abs[z,jh,jd] for z in Z, jh in JH, jd in JD] #Change
P_supvec = [P_sup[z,jh,jd] for z in Z, jh in JH, jd in JD] #Change


# Select day for which you'd like to plotting
jd = 1

# Electricity price 
p1 = plot(JH,λvec[:,jd], xlabel="Timesteps [-]", ylabel="λ [EUR/MWh]", label="λ [EUR/MWh]", legend=:outertopright );

# Dispatch
p2 = groupedbar(transpose(gvec[:,:,jd]), label=["Mid" "Base" "Peak" "Wind" "Solar"], bar_position = :stack,legend=:outertopright,ylims=(0,13_000));
plot!(p2, JH, D[:,jd], label ="Demand", xlabel="Timesteps [-]", ylabel="Generation [MWh]", legend=:outertopright, lindewidth=3, lc=:black);

# Capacity
p3 = bar(capvec, label="", xticks=(1:length(capvec), ["Mid" "Base" "Peak" "Wind" "Solar"]), xlabel="Technology [-]", ylabel="New capacity [MW]", legend=:outertopright);

# ESS
p4 =  bar(ESSvec, label="", xticks=(1:length(ESSvec), ["PSH" "BESS"]),xlabel="Type of ESS [-]", ylabel="State of ESS [MW]", legend=:outertopright);

# combine
plot(p1, p2, p3, p4, layout = (4,1))
plot!(size=(1500,1500))