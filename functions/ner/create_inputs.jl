using Transformers.HuggingFace
using JLD2
using TextAnalysis
using Random


greek_tokenizer = hgf"nlpaueb/bert-base-greek-uncased-v1:tokenizer"

include("ner.jl")
include("./../preprocess.jl")


# Inputs
NoDocs = parse(Int, ARGS[1])
overlappingChunks = parse(Int, ARGS[2]) 
set = parse(Int, ARGS[3])

mainDir = "./results/$NoDocs/"

if !isdir(mainDir)
    mkdir(mainDir)
end

SIDLabels = ["No", "Yes"]

if isfile(mainDir*"trainingSet.jld2")
    training = JLD2.load(mainDir*"trainingSet.jld2")
    validation = JLD2.load(mainDir*"valSet.jld2")
    test = JLD2.load(mainDir*"testSet.jld2")

    if set == 1 
        ner_train = find_ner(training["crps"], overlappingChunks)
        save(mainDir * "ner_train.jld2", Dict("ner_train" => ner_train))
    elseif set == 2
        ner_val = find_ner(validation["crps"], overlappingChunks)
        save(mainDir * "ner_val.jld2", Dict("ner_val" => ner_val))
    elseif set == 3
        ner_test = find_ner(test["crps"], overlappingChunks)
        save(mainDir * "ner_test.jld2", Dict("ner_test" => ner_test))
    end
else 
    corpus = load("./data/outputs/sampleCorpus.jld2", "crps")
    targets = load("./data/outputs/SIDTargets.jld2", "trgs")

    corpus = corpus[1:NoDocs]
    targets = targets[1:NoDocs]

    training, validation, test = split_data(corpus, targets)
    save(mainDir * "trainingSet.jld2", Dict("crps"=>training[1], "targets"=>training[2]))
    save(mainDir * "valSet.jld2", Dict("crps"=>validation[1], "targets"=>validation[2]))
    save(mainDir * "testSet.jld2", Dict("crps"=>test[1], "targets"=>test[2]))

    if set == 1 
        ner_train = find_ner(training[1], overlappingChunks)
        save(mainDir * "ner_train.jld2", Dict("ner_train" => ner_train))
    elseif set == 2
        ner_val = find_ner(validation[1], overlappingChunks)
        save(mainDir * "ner_val.jld2", Dict("ner_val" => ner_val))
    elseif set == 3
        ner_test = find_ner(test[1], overlappingChunks)
        save(mainDir * "ner_test.jld2", Dict("ner_test" => ner_test))
    end
end