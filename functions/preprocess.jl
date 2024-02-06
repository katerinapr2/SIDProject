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


function inputs(docs, targets, SIDLabels, greek_tokenizer, overlappingChunks)
    allChunks = Vector{Vector{String}}()
    labels = Vector{Vector{String}}()
    for z in zip(docs, targets)
        if overlappingChunks == 1
            chunks, lenChunks, sI, eI = ToChunks.split_to_overlapping_chunks(z[1].text, greek_tokenizer)
        else
            chunks, lenChunks, sI, eI = ToChunks.split_to_chunks(z[1].text, greek_tokenizer)
        end

        if typeof(chunks) == String 
            push!(allChunks, [chunks])
            push!(labels, split(z[2].text))
        else
            for (chunk, s, e) in zip(chunks, sI, eI)
                push!(allChunks, [chunk])
                push!(labels, split(z[2].text)[s:e])
            end
        end

    end

    enc = Transformers.encode(greek_tokenizer, allChunks)
    # tokens = Transformers.decode(greek_tokenizer, enc.token)

    labelsDict = Dict()
    for i in 1:length(SIDLabels)
        labelsDict[SIDLabels[i]] = i
    end

    targets = []
    for i in 1:length(labels)
        temp = []
        for j in 1:length(labels[i])
                push!(temp, labelsDict[labels[i][j]] - 1)
        end
        push!(targets, temp)
    end

    return enc, targets
end


function preprocessDoc(doc, greek_tokenizer)
    allChunks = Vector{Vector{String}}()
    
    chunks, lenChunks, _ = ToChunks.split_to_chunks(doc, greek_tokenizer)
    if typeof(chunks) == String 
        push!(allChunks, [chunks])
    else
        for (chunk, len) in zip(chunks, lenChunks)
            push!(allChunks, [chunk])
        end
    end

    enc = Transformers.encode(greek_tokenizer, allChunks)
    tokens = Transformers.decode(greek_tokenizer, enc.token)
    return enc, tokens
end

