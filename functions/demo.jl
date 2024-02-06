using Flux.Data: DataLoader
using JLD2
using Random
using CUDA
using OneHotArrays
using StatsBase: mean
using BSON: @save
using JSON
using Plots
using TextAnalysis
using Transformers.HuggingFace

using SIDProject

include("toChunks.jl")
using .ToChunks
include("preprocess.jl")
include("training.jl")

Random.seed!(123)

enable_gpu()
GC.gc(true)


## Split corpus to training, validation and test sets
function split_data(corpus, targets)
    shuffledIndex = shuffle!(Vector(1:length(corpus)))

    index1 = round(Int, 0.8*length(shuffledIndex))
    index2 = round(Int, 0.1*length(shuffledIndex))

    if length(corpus) < 10 
        index1 -= 1
        index2 += 1
    end

    trainingIndex = shuffledIndex[1:index1]
    validationIndex = shuffledIndex[index1+1:index1+index2]
    testIndex = shuffledIndex[index1+index2+1:end]

    training = corpus[trainingIndex], targets[trainingIndex]
    validation = corpus[validationIndex], targets[validationIndex]
    test = corpus[testIndex], targets[testIndex]
    return training, validation, test
end



# Inputs 
epochs = parse(Int, ARGS[1])
NoDocs = parse(Int, ARGS[2])
BatchSize = parse(Int, ARGS[3])
lr = parse(Float64, ARGS[4])
d = parse(Float64, ARGS[5])
overlappingChunks = parse(Int, ARGS[6]) 


mainDir = "results\\$NoDocs\\"
dir = mainDir * "bs$BatchSize\\lr$lr\\d$d\\e$epochs\\chunks$overlappingChunks\\"

if !isdir(dir)
    mkpath(dir)
end

SIDLabels = ["No", "Yes"]

greek_tokenizer = hgf"nlpaueb/bert-base-greek-uncased-v1:tokenizer"


if isfile(mainDir*"trainingSet.jld2")
    training = JLD2.load(mainDir*"trainingSet.jld2")
    validation = JLD2.load(mainDir*"valSet.jld2")
    test = JLD2.load(mainDir*"testSet.jld2")

    X_train, y_train = inputs(training["crps"], training["targets"], SIDLabels, greek_tokenizer, overlappingChunks)
    X_val, y_val = inputs(validation["crps"], validation["targets"], SIDLabels, greek_tokenizer, overlappingChunks)
    X_test, y_test = inputs(test["crps"], test["targets"], SIDLabels, greek_tokenizer, 0)

else
    corpus = load(".\\..\\thesis\\anonymization\\data\\sampleCorpus_3898.jld2", "crps")
    targets = load(".\\..\\thesis\\anonymization\\data\\SIDTargets_3898.jld2", "trgs")

    corpus = corpus[1:NoDocs]
    targets = targets[1:NoDocs]

    training, validation, test = split_data(corpus, targets)
    save(mainDir*"trainingSet.jld2", Dict("crps"=>training[1], "targets"=>training[2]))
    save(mainDir*"valSet.jld2", Dict("crps"=>validation[1], "targets"=>validation[2]))
    save(mainDir*"testSet.jld2", Dict("crps"=>test[1], "targets"=>test[2]))

    X_train, y_train = inputs(training[1], training[2], SIDLabels, greek_tokenizer, overlappingChunks)
    X_val, y_val = inputs(validation[1], validation[2], SIDLabels, greek_tokenizer, overlappingChunks)
    X_test, y_test = inputs(test[1], test[2], SIDLabels, greek_tokenizer, 0)
end


train_data = (X_train.token, X_train.segment, X_train.attention_mask.len, y_train)
val_data = (X_val.token, X_val.segment, X_val.attention_mask.len, y_val)
test_data = (X_test.token, X_test.segment, X_test.attention_mask.len, y_test)

train_data_loader = DataLoader(train_data; batchsize=BatchSize, shuffle=false)
val_data_loader = DataLoader(val_data; batchsize=BatchSize, shuffle=false)
test_data_loader = DataLoader(test_data; batchsize=BatchSize, shuffle=false)


# Trainig and Testing of the model
# Results

model = create_model_withoutNER(length(SIDLabels), d) |> gpu 
opt = Adam(lr);

history = trainModel!(model, train_data_loader, opt, val_data_loader, SIDLabels, dir; n_epochs=epochs)


## Plots
epochs = 1:epochs
p1 = plot(epochs, [history["train_acc"] history["val_acc"]], legend=false)
xlabel!("epochs")
ylabel!("Accuracy")

p2 = plot(epochs, [history["train_loss"] history["val_loss"]], label=["training" "validation"])
xlabel!("epochs")
ylabel!("Loss")

p3 = plot(epochs, [history["train_prec"] history["val_prec"]], legend=false)
xlabel!("epochs")
ylabel!("Precision")

p4 = plot(epochs, [history["train_rec"] history["val_rec"]], legend=false)
xlabel!("epochs")
ylabel!("Recall")

p = plot(p1, p2, p3, p4, layout=(2,2))
savefig(p, dir*"history.png")

testing(model, test_data_loader, SIDLabels, dir)