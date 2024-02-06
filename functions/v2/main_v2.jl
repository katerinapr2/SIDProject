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

include(".\\..\\preprocess.jl")
include("training_v2.jl")
include("filterSets_withNER.jl")

Random.seed!(123)

enable_gpu()
GC.gc(true)



function dimCompatibility(x, y, n, ft, fi)
    indx = []
    for i in 1:length(n)
        if x.attention_mask.len[i] != (length(n[i])+2)
            push!(indx, i)
        end
    end

    if !isempty(indx)
        deleteat!(n, indx)
        deleteat!(ft, indx)
        deleteat!(fi, indx)
        deleteat!(y, indx)

        t = x.token[:,:,setdiff(1:end, indx)]
        s = x.segment[:,setdiff(1:end, indx)]
        a = x.attention_mask.len[setdiff(1:end, indx)]

        return (token=t, segment=s, attention_mask=LengthMask{1, Vector{Int32}}(a)), y, n, ft, fi, indx
    else
        return x, y, n, ft, fi, indx
    end
end


# Inputs 
epochs = parse(Int, ARGS[1])
NoDocs = parse(Int, ARGS[2])
BatchSize = parse(Int, ARGS[3])
lr = parse(Float64, ARGS[4])
d = parse(Float64, ARGS[5])
overlappingChunks = parse(Int, ARGS[6]) 


mainDir = "results\\$NoDocs\\"
dir = mainDir * "v2\\bs$BatchSize\\lr$lr\\d$d\\e$epochs\\chunks$overlappingChunks\\"

if !isdir(dir)
    mkpath(dir)
end

SIDLabels = ["No", "Yes"]

greek_tokenizer = hgf"nlpaueb/bert-base-greek-uncased-v1:tokenizer"
# NER = gr_nlp_pipeline.Pipeline("ner")

println("Sets: loading...")

if isfile(mainDir*"trainingSet.jld2")
    training = JLD2.load(mainDir*"trainingSet.jld2")
    validation = JLD2.load(mainDir*"valSet.jld2")
    test = JLD2.load(mainDir*"testSet.jld2")

    X_train, y_train = inputs(training["crps"], training["targets"], SIDLabels, greek_tokenizer, overlappingChunks)
    X_val, y_val = inputs(validation["crps"], validation["targets"], SIDLabels, greek_tokenizer, overlappingChunks)
    X_test, y_test = inputs(test["crps"], test["targets"], SIDLabels, greek_tokenizer, overlappingChunks)
else
    corpus = load(".\\data\\outputs\\sampleCorpus.jld2", "crps")
    targets = load(".\\data\\outputs\\SIDTargets.jld2", "trgs")

    corpus = corpus[1:NoDocs]
    targets = targets[1:NoDocs]

    training, validation, test = split_data(corpus, targets)
    save(mainDir*"trainingSet.jld2", Dict("crps"=>training[1], "targets"=>training[2]))
    save(mainDir*"valSet.jld2", Dict("crps"=>validation[1], "targets"=>validation[2]))
    save(mainDir*"testSet.jld2", Dict("crps"=>test[1], "targets"=>test[2]))

    X_train, y_train = inputs(training[1], training[2], SIDLabels, greek_tokenizer, overlappingChunks)
    X_val, y_val = inputs(validation[1], validation[2], SIDLabels, greek_tokenizer, overlappingChunks)
    X_test, y_test = inputs(test[1], test[2], SIDLabels, greek_tokenizer, overlappingChunks)
end

println("NER Labels: loading...")

# if !isfile(mainDir*"ner_train.jld2")
#     ner_train = find_ner(training["crps"], greek_tokenizer, NER, overlappingChunks)
#     save(mainDir * "ner_train.jld2", Dict("ner_train" => ner_train))
# else
    ner_train = JLD2.load(mainDir * "ner_train.jld2")["ner_train"] # without special tokens
# end

# if !isfile(mainDir*"ner_val.jld2")
#     ner_val = find_ner(validation["crps"], greek_tokenizer, NER, overlappingChunks)
#     save(mainDir * "ner_val.jld2", Dict("ner_val" => ner_val))
# else
    ner_val = JLD2.load(mainDir * "ner_val.jld2")["ner_val"]
# end

# if !isfile(mainDir*"ner_test.jld2")
#     ner_test = find_ner(test["crps"], greek_tokenizer, NER, overlappingChunks)
#     save(mainDir * "ner_test.jld2", Dict("ner_test" => ner_test))
# else
    ner_test = JLD2.load(mainDir * "ner_test.jld2")["ner_test"]
# end



println("Filtered Sets: loading...")

if !isfile(mainDir*"filteredTrainSet.jld2")
    # indexes_train = []
    # for ch in 1:length(ner_train)
    #     push!(indexes_train, findall(x->(occursin("PERSON", x) | occursin("LOC", x) | occursin("GPE", x)), ner_train[ch]))
    # end

    # y_train_filtered = filter_labels(indexes_train, y_train)
    y_train_filtered, indexes_train = createFilteredSet(y_train, ner_train)
    ftrain = Dict("filtered_targets" => y_train_filtered, "indexes" => indexes_train)
    save(mainDir * "filteredTrainSet.jld2", ftrain)
else
    ftrain = JLD2.load(mainDir*"filteredTrainSet.jld2")
end

if !isfile(mainDir*"filteredValSet.jld2")
    # indexes_val = []
    # for y in y_val
    #     push!(indexes_val, [i for i in 1:length(y)])
    # end
    
    
    # indexes_val = []
    # for ch in 1:length(ner_val)
    #     push!(indexes_val, findall(x->(occursin("PERSON", x) | occursin("LOC", x) | occursin("GPE", x)), ner_val[ch]))
    # end

    # y_val_filtered = filter_labels(indexes_val, y_val)

    y_val_filtered, indexes_val = createFilteredSet(y_val, ner_val)
    fval = Dict("filtered_targets" => y_val_filtered, "indexes" => indexes_val)
    save(mainDir * "filteredValSet.jld2", fval)
else
    fval = JLD2.load(mainDir*"filteredValSet.jld2")
end

if !isfile(mainDir*"filteredTestSet.jld2")
    # indexes_test = []
    # for y in y_test
    #     push!(indexes_test, [i for i in 1:length(y)])
    # end
    
    # indexes_test = []
    # for ch in 1:length(ner_test)
    #     push!(indexes_test, findall(x->(occursin("PERSON", x) | occursin("LOC", x) | occursin("GPE", x)), ner_test[ch]))
    # end

    # y_test_filtered = filter_labels(indexes_test, y_test)
    y_test_filtered, indexes_test = createFilteredSet(y_test, ner_test)
    ftest = Dict("filtered_targets" => y_test_filtered, "indexes" => indexes_test)
    save(mainDir * "filteredTestSet.jld2", ftest)
else
    ftest = JLD2.load(mainDir*"filteredTestSet.jld2")
end

println("$(length(X_train.attention_mask.len)) --- $(length(ner_train))")
X_train, y_train, ner_train, ft_train, fi_train, out = dimCompatibility(X_train, y_train, ner_train, ftrain["filtered_targets"], ftrain["indexes"])
println("$(length(out)) removed.")
X_val, y_val, ner_val, ft_val, fi_val, out = dimCompatibility(X_val, y_val, ner_val, fval["filtered_targets"], fval["indexes"])
X_test, y_test, ner_test, ft_test, fi_test, out = dimCompatibility(X_test, y_test, ner_test, ftest["filtered_targets"], ftest["indexes"])

train_data = (X_train.token, X_train.segment, X_train.attention_mask.len, ft_train, fi_train)
val_data = (X_val.token, X_val.segment, X_val.attention_mask.len, ft_val, fi_val)
test_data = (X_test.token, X_test.segment, X_test.attention_mask.len, ft_test, fi_test)

train_data_loader = DataLoader(train_data; batchsize=BatchSize, shuffle=false)
val_data_loader = DataLoader(val_data; batchsize=BatchSize, shuffle=false)
test_data_loader = DataLoader(test_data; batchsize=BatchSize, shuffle=false)


# Trainig and Testing of the model
# Results

model = create_model(length(SIDLabels), d) |> gpu 
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