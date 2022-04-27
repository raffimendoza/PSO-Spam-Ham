using CSV, DataFrames, Plots, PackageCompiler
using Statistics, StatsBase
using TextAnalysis: NaiveBayesClassifier, fit!, predict
# --References-- #
# https://www.kaggle.com/uciml/sms-spam-collection-dataset?select=spam.csv
# https://medium.com/@sddkal/julia-generic-particle-swarm-optimization-algoritmas%C4%B1-ger%C3%A7eklemesi-80a23371abb4
# Ham or Spam classification

# HW5

spamdata = CSV.read("spam.csv",DataFrame)
train_population = 3901
test_population = train_population+1672
training_particles = []
test_particles = []

# Classify ham/spam in list from data set
hamList = []
spamList =  []
unionList = []
for row in eachrow(spamdata)
    push!(unionList, filter(isvalid, row.v2), " ")
    if row.v1 == "ham"
        push!(hamList, filter(isvalid, row.v2), " ")
    elseif  row.v1 == "spam"
        push!(spamList, filter(isvalid, row.v2), " ")
    end
end
println("Finished classifying spam/ham")

# Use join function to form ham/spam list into one string
hamString = join(hamList)
spamString = join(spamList)

# Seperate ham/spam string into words and create dicionary Dict(Word,Count) to be the updating weight for the fitness function; Count = amount of weight the word impacts e.g. A spam word higher than 2 will be considered a spam word
hamDict = countmap(split(replace(hamString, '.' => "")," "))
hamDict = sort(hamDict, byvalue=true, rev=true)
println("Ham: ", length(hamDict))
spamDict = countmap(split(replace(spamString, '.' => "")," "))
spamDict = sort(spamDict, byvalue=true, rev=true)
println("Spam: ", length(spamDict))
println("Finished creating dictionary spam/ham")

function funct(x)
    return (1 - x[1]) ^ 2 + 100 * (x[2] - x[1] ^ 2) ^ 2
end

struct Particle{T};
    x::Vector{T};
    v::Vector{T};
    fit::String;
end


struct PSO{T<:Real, N};
    particles::Array{Particle{T}, 1}; 
    fitness::Function; 
    particle_count::Int64;
end

function fitVal(msg)
    fitness = 0
    count = 0
    ch = countmap(split(replace(msg, '.' => "")," "))
    for word in ch
        if haskey(spamDict, word.first) && haskey(hamDict, word.first)
            spamDict[word.first] = 0 # Remove duplicate count from spam dictionary -> less likely to be a spam word
            hamDict[word.first] = hamDict[word.first] + 1 # Add 1 to ham dictionary value to increase fitness for next gen
            fitness = fitness + 1*(hamDict[word.first]/10000)
            count = count + 1
        elseif haskey(hamDict, word.first) && (hamDict[word.first] > 2 && hamDict[word.first] < 1000)
            hamDict[word.first] = hamDict[word.first] + 1 # Add 1 to ham dictionary value to increase fitness for next gen
            fitness = fitness + 1*(hamDict[word.first]/10000)
            count = count + 1
        elseif haskey(spamDict, word.first) && (spamDict[word.first] > 2 && spamDict[word.first] < 1000)
            spamDict[word.first] = spamDict[word.first] + 1 # Add 1 to spam dictionary to decrease fitness
            fitness = fitness + 0
            count = count + 1
        else
            push!(hamDict, String(word.first) => 1)
            fitness = fitness + 0
            count = count + 1
        end
    end
    return (fitness/count)
end


function PSO{T, N}(fitness, particle_count) where {T <: Real, N}
    obj = new(Array{Particle{T}, 1}(UndefInitializer(),
     particle_count), fitness, particle_count);
    for i in 1:particle_count
        obj.particles[i] = Particle{T}(Vector{T}(rand(N)),
         Vector{T}(zeros(N)))
    end
    obj # return
end

# Update v
function update_particle_v(p::Particle, offset)
    for i in 1:length(p.x)
        p.v[i] += offset[i]
    end
end

# Update x
function update_particle_x(p::Particle)
    for i in 1:length(p.x)
        p.x[i] += p.v[i]
    end
end

# Initialize PSO
function optimize_pos(pos::PSO, max_iter::Int64 = 10, c1::Float64 = 0.1, c2::Float64 = 0.3, type::String = "train")
    l = length(pos.particles)
    #distances = zeros(l)
    fitnessEvals = zeros(l)
    gbest_val = typemax(Float64) 
    gbest = nothing 
    for i in 1:max_iter
        for (i, p) in enumerate(pos.particles)
            update_particle_x(p)
            fitnessEvals[i] = pos.fitness(p.fit)
            println("Particle fitness evaluation: ", fitnessEvals[i], "\n") 
        end
        best_index = argmax(fitnessEvals)
        println("Best index: ", best_index)
        if type == "train"
            push!(training_particles, maximum(fitnessEvals))
        elseif type == "test"
            push!(test_particles, maximum(fitnessEvals))
        end
        pbest = pos.particles[best_index]
        if gbest_val > fitnessEvals[best_index]
            gbest = deepcopy(pbest) 
            gbest_val = fitnessEvals[best_index]
        end
        for p in pos.particles
            offset = c1 * rand()*(pbest.x - p.x) + c2 * rand()*(gbest.x - p.x)
            update_particle_v(p, offset)
        end
    end
    println("GBest Val:", gbest_val)
    gbest # return
end

half = Integer.((length(unionList)/2))

hiddenLayer = []
hiddenPop = 2000
for i in 1:hiddenPop
    push!(hiddenLayer, unionList[rand(1:1:length(unionList))])
end

spamLayer = 1000
hamLayer = 500

# Train Data
arr = Array{Particle, 1}(undef, half)

for i in 1:half
    arr[i] = Particle(Float32.(rand(3)),Float32.(rand(3)),unionList[i])
end
for i in 1:hiddenPop
    push!(arr, Particle(Float32.(rand(3)),Float32.(rand(3)),hiddenLayer[i]))
end 
for i in spamLayer
    push!(arr, Particle(Float32.(rand(3)),Float32.(rand(3)),spamList[rand(1:length(spamList))]))
end
for i in hamLayer
    push!(arr, Particle(Float32.(rand(3)),Float32.(rand(3)),spamList[rand(1:length(hamLayer))]))
end

n_particles = 20
pso_instance = PSO{Float32, 2}(arr, fitVal, n_particles)
#println(pso_instance)
println("Optimized train results: ", optimize_pos(pso_instance, 10))

# Test Data
arr2 = Array{Particle, 1}(undef, half)
count = 1
for i in half:length(unionList)-1
    arr2[count] = Particle(Float32.(rand(3)),Float32.(rand(3)),unionList[i])
    global count = count + 1
end
for i in 1:hiddenPop
    push!(arr2, Particle(Float32.(rand(3)),Float32.(rand(3)),hiddenLayer[i]))
end
for i in spamLayer
    push!(arr2, Particle(Float32.(rand(3)),Float32.(rand(3)),spamList[rand(1:length(spamList))]))
end
for i in hamLayer
    push!(arr2, Particle(Float32.(rand(3)),Float32.(rand(3)),spamList[rand(1:length(hamLayer))]))
end

n_particles = 20
pso_instance = PSO{Float32, 2}(arr2, fitVal, n_particles)
#println(pso_instance)
println("Optimized test results: ", optimize_pos(pso_instance,10, 0.1, 0.3,"test"))

# Plot
println()
p = plot([1:length(training_particles)], training_particles, title = "Results", label=["Training"])
plot!(p, [1:length(training_particles)], test_particles, label=["Testing"])
savefig("plot.png")
println("Exported plot.")