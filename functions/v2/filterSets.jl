# using JLD2
# using TextAnalysis
# using Transformers.HuggingFace
# using BSON: @save
# using Random

# include("./../preprocess.jl")



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


# function addExtras(sensInd, nonSensInd, c0, c1)
#    while c1 - c0 >= 510 
#         temp = findall(x->!isempty(x), nonSensInd)
#         ch = temp[rand(1:length(temp))]
  
#         i = rand(1:length(nonSensInd[ch]))
#         shuffle!(nonSensInd[ch])
#         append!(sensInd[ch], nonSensInd[ch][1:i])
#         deleteat!(nonSensInd[ch], 1:i)
#         c0 += i
#    end

#    while c1 - c0 != 0
#         temp = findall(x->!isempty(x), nonSensInd)
#         ch = temp[rand(1:length(temp))]
   
#         if (c1 - c0) <= length(nonSensInd[ch])
#             append!(sensInd[ch], nonSensInd[ch][1:(c1-c0)])
#             deleteat!(nonSensInd[ch], 1:(c1-c0))
#             c0 += (c1-c0)
#         else
#             append!(sensInd[ch], nonSensInd[ch])
#             c0 += length(nonSensInd[ch])
#             nonSensInd[ch] = []
#         end
#     end

#     return sensInd, nonSensInd, c0, c1
# end

function addExtras(sensInd, nonSensInd, c0, c1)
    for ch in 1:length(sensInd)
        shuffle!(nonSensInd[ch])
        extras = length(sensInd[ch])
        append!(sensInd[ch], nonSensInd[ch][1:extras])
        shuffle!(sensInd[ch])
        deleteat!(nonSensInd[ch], 1:extras)
        c0 += extras
    end
    return sensInd, nonSensInd, c0, c1
end


function createFilteredSet(y)
    sens, nonSens = separateIndexes(y)
    cns = 0
    cs = sum([length(s) for s in sens])

    indexes, nonSens, cns, cs = addExtras(sens, nonSens, cns, cs)
    y_filtered = filter_labels(indexes, y)
    return y_filtered, indexes
end


# NoDocs = parse(Int, ARGS[1])
# overlappingChunks = parse(Int, ARGS[2]) 
# set = parse(Int, ARGS[3])

# mainDir = "./results/$NoDocs/"

# SIDLabels = ["No", "Yes"]

# greek_tokenizer = hgf"nlpaueb/bert-base-greek-uncased-v1:tokenizer"

# if isfile(mainDir * "trainingSet.jld2")
#     if set == 1 
#         training = JLD2.load(mainDir * "trainingSet.jld2")
#         _, y_train = inputs(training["crps"], training["targets"], SIDLabels, greek_tokenizer, overlappingChunks)
        
#         sens_train, nonSens_train = separateIndexes(y_train)

#         cns_train = 0
#         cs_train = sum([length(s) for s in sens_train])

#         indexes_train, nonSens_train, cns_train, cs_train = addExtras(sens_train, nonSens_train, cns_train, cs_train)
#         y_train_filtered = filter_labels(indexes_train, y_train)
#         save(mainDir * "filteredTrainSet.jld2", Dict("filtered_targets" => y_train_filtered, "indexes" => indexes_train))
#     elseif set == 2
#         validation = JLD2.load(mainDir * "valSet.jld2")
#         X_val, y_val = inputs(validation["crps"], validation["targets"], SIDLabels, greek_tokenizer, overlappingChunks)

#         sens_val, nonSens_val = separateIndexes(y_val)

#         cns_val = 0
#         cs_val = sum([length(s) for s in sens_val])

#         indexes_val, nonSens_val, cns_val, cs_val = addExtras(sens_val, nonSens_val, cns_val, cs_val)
#         y_val_filtered = filter_labels(indexes_val, y_val)
#         save(mainDir * "filteredValSet.jld2", Dict("filtered_targets" => y_val_filtered, "indexes" => indexes_val))
#     elseif set == 3
#         test = JLD2.load(mainDir * "testSet.jld2")
#         _, y_test = inputs(test["crps"], test["targets"], SIDLabels, greek_tokenizer, 0)
        
#         indexes_test = []
#         for i in y_test
#             push!(indexes_test, [i for i in 1:length(i)])
#         end
 
#         # sens_test, nonSens_test = separateIndexes(y_test)

#         # cns_test = 0
#         # cs_test = sum([length(s) for s in sens_test])

#         # indexes_test, nonSens_test, cns_test, cs_test = addExtras(sens_test, nonSens_test, cns_test, cs_test)
#         # y_test_filtered = filter_labels(indexes_test, y_test)
#         save(mainDir * "filteredTestSet.jld2", Dict("filtered_targets" => y_test, "indexes" => indexes_test))
#     end

# end



