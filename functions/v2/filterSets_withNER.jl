using JLD2
using TextAnalysis
using Transformers.HuggingFace
using BSON: @save
using Random

include("./../preprocess.jl")

export createFilteredSet, filter_labels



function separateIndexes(y)
    sensIndexes = []
    nonSensIndexes = []
    for n in 1:length(y)
        push!(sensIndexes, findall(x->x==1, y[n]))
        push!(nonSensIndexes, findall(x->x==0, y[n]))
    end
    return sensIndexes, nonSensIndexes
end


function filter_labels(index, y)
    filtered = []
    for i in 1:length(index)
        push!(filtered, y[i][index[i]])
    end
    return filtered
end


# function addExtras(y_temp, sensInd, nonSensInd, c0, c1)
#     s, ns = separateIndexes(y_temp[ch])
#     for ch in 1:length(sensInd)
#         shuffle!(nonSensInd[ch])
#         # extras = length(findall(x->x==1, y_temp[ch])) - length(findall(x->x==0, y_temp[ch])) #length(sensInd[ch])
#         extras = length(s[ch]) - length(ns[ch])
#         if extras > 0 
#             append!(sensInd[ch], nonSensInd[ch][1:extras])
#             shuffle!(sensInd[ch])
#             deleteat!(nonSensInd[ch], 1:extras)
#             c0 += extras
#         end
#     end
#     return sensInd, nonSensInd, c0, c1
# end


# function countSamples(arr, target)
#     return length(findall(x->x==target, arr))
# end


function addExtras(sensInd, nonSensInd, c0, c1)
   while c1 - c0 >= 510 
        temp = findall(x->!isempty(x), nonSensInd)
        ch = temp[rand(1:length(temp))]
  
        i = rand(1:length(nonSensInd[ch]))
        shuffle!(nonSensInd[ch])
        append!(sensInd[ch], nonSensInd[ch][1:i])
        shuffle!(sensInd[ch])
        deleteat!(nonSensInd[ch], 1:i)
        c0 += i
   end

   while c1 - c0 != 0
        temp = findall(x->!isempty(x), nonSensInd)
        ch = temp[rand(1:length(temp))]
   
        if (c1 - c0) <= length(nonSensInd[ch])
            append!(sensInd[ch], nonSensInd[ch][1:(c1-c0)])
            shuffle!(sensInd[ch])
            deleteat!(nonSensInd[ch], 1:(c1-c0))
            c0 += (c1-c0)
        else
            append!(sensInd[ch], nonSensInd[ch])
            shuffle!(sensInd[ch])
            c0 += length(nonSensInd[ch])
            nonSensInd[ch] = []
        end
    end

    return sensInd, nonSensInd, c0, c1
end


function createFilteredSet(y, nerLabels)
    sens, nonSens = separateIndexes(y)

    nonSens_ner = []
    for ch in 1:length(nerLabels)
        nerInds = findall(x->(occursin("PERSON", x) | occursin("LOC", x) | occursin("GPE", x)), nerLabels[ch])
        
        # keep only the non sensitive ner samples 
        try
            push!(nonSens_ner, nerInds[findall(x->y[ch][x]==0, nerInds)])
        catch
            println("$ch")
            break
        end
    end
    ## add the non sensitive ner samples to the array with 
    # sensitive samples and delete them from the array 
    # with non sens. samples. 
    ## count the samples of each class that they are contained 
    # on sens array
    cs, cns = 0, 0
    for ch in 1:length(nonSens)
        cs += length(sens[ch])
        append!(sens[ch], nonSens_ner[ch])
        deleteat!(nonSens[ch], findall(x->(x in nonSens_ner[ch]), nonSens[ch]))
        cns += length(nonSens_ner[ch])
    end

    ## add extra non sensitive samples for balanced dataset
    indexes, nonSens, cns, cs = addExtras(sens, nonSens, cns, cs)
    return filter_labels(indexes, y), indexes
end



NoDocs = parse(Int, ARGS[1])
overlappingChunks = parse(Int, ARGS[2]) 
set = parse(Int, ARGS[3])


mainDir = "./results/$NoDocs/"
# mainDir = "./../thesis/anonymization/data/"

SIDLabels = ["No", "Yes"]

greek_tokenizer = hgf"nlpaueb/bert-base-greek-uncased-v1:tokenizer"

# if isfile(mainDir * "trainingSet.jld2")
    if set == 1 
        training = JLD2.load(mainDir * "trainingSet.jld2")
        ner_train = JLD2.load(mainDir * "ner_train.jld2")
        _, y_train = inputs(training["crps"], training["targets"], SIDLabels, greek_tokenizer, overlappingChunks)

        y_train_filtered, indexes_train = createFilteredSet(y_train, ner_train["ner_train"])
        save(mainDir * "filteredTrainSet.jld2", Dict("filtered_targets" => y_train_filtered, "indexes" => indexes_train))
    elseif set == 2
        validation = JLD2.load(mainDir * "valSet.jld2")
        # valCorpus = JLD2.load(mainDir * "valCorpus.jld2", "crps").documents
        # valTargets = JLD2.load(mainDir * "valSIDTargets.jld2", "trgs").documents
        ner_val = JLD2.load(mainDir * "ner_val.jld2")
        _, y_val = inputs(validation["crps"], validation["targets"], SIDLabels, greek_tokenizer, overlappingChunks)

        y_val_filtered, indexes_val = createFilteredSet(y_val, ner_val["ner_val"])
        save(mainDir * "filteredValSet.jld2", Dict("filtered_targets" => y_val_filtered, "indexes" => indexes_val))
    elseif set == 3
        test = JLD2.load(mainDir * "testSet.jld2")
        ner_test = JLD2.load(mainDir * "ner_test.jld2")
        _, y_test = inputs(test["crps"], test["targets"], SIDLabels, greek_tokenizer, overlappingChunks)

        # indexes_test = []
        # for y in y_test
        #     push!(indexes_test, [i for i in 1:length(y)])
        # end
        # save(mainDir * "filteredTestSet.jld2", Dict("filtered_targets" => y_test, "indexes" => indexes_test))

        y_test_filtered, indexes_test = createFilteredSet(y_test, ner_test["ner_test"])
        save(mainDir * "filteredTestSet.jld2", Dict("filtered_targets" => y_test_filtered, "indexes" => indexes_test))
    end
# end
        