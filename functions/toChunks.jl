## Manage Long Texts
## Output: the non overlapping chunks of the text and the length of each one 

module ToChunks

using TextEncodeBase

export split_to_chunks, split_to_overlapping_chunks

function for_first_token(tokens, startIndex, endIndex)
    while startswith(tokens[startIndex].x, "##")
        startIndex = startIndex + 1
        if endIndex <length(tokens)
            endIndex = endIndex + 1
        end
    end
    return startIndex, endIndex
end


function for_last_token(tokens, endIndex)
    while startswith(tokens[endIndex+1].x, "##")
        endIndex = endIndex - 1
    end
    return endIndex
end


function split_to_chunks(doc, tokenizer)

    tokens = TextEncodeBase.tokenize(tokenizer, doc)
    
    if length(tokens) > 510
        temp = ""
        chunks = Vector{String}()
        lenChunks = Vector{Int}()
        sI = Vector{Int}()
        eI = Vector{Int}()
        startIndex = 1
        endIndex = 510  # 2 tokens are the [CLS] and [SEP].
        while endIndex < length(tokens)
            (startIndex, endIndex) = for_first_token(tokens, startIndex, endIndex)
            endIndex = for_last_token(tokens, endIndex)
            
            # println("\n\nstart: " * string(startIndex) * " - end: " * string(endIndex) * '\n')

            for token in tokens[startIndex:endIndex]
                temp = occursin("##", token.x) ? (temp * replace(token.x, "##"=>"")) : (temp * ' ' * token.x)
            end
            push!(chunks, temp)
            push!(lenChunks, length(tokens[startIndex:endIndex]))
            push!(sI, startIndex)
            push!(eI, endIndex)
            temp = ""

            startIndex = endIndex + 1
            endIndex = length(tokens) > (startIndex + 509) ? (startIndex + 509) : length(tokens)
        end

        # println("\n\nstart: " * string(startIndex) * " - end: " * string(endIndex) * '\n')
        # for i in eachindex(temp[end])
        #     print(temp[end][i].x * ' ')
        # end
        # print("\n\n")

        for token in tokens[startIndex:endIndex]
            temp = occursin("##", token.x) ? (temp * replace(token.x, "##"=>"")) : (temp * ' ' * token.x)
        end
        push!(chunks, temp)
        push!(lenChunks, length(tokens[startIndex:endIndex]))
        push!(sI, startIndex)
        push!(eI, endIndex)

        return chunks, lenChunks, sI, eI;
    else
        return doc, length(tokens), 1, length(tokens);
    end
end

function split_to_overlapping_chunks(doc, tokenizer)
    tokens = TextEncodeBase.tokenize(tokenizer, doc)
    
    if length(tokens) > 510
        temp = ""
        chunks = Vector{String}()
        lenChunks = Vector{Int}()
        sI = Vector{Int}()
        eI = Vector{Int}()
        startIndex = 1
        endIndex = 510  # 2 tokens are the [CLS] and [SEP].
        while endIndex < length(tokens)
            (startIndex, endIndex) = for_first_token(tokens, startIndex, endIndex)
            
            if endIndex != length(tokens)
                endIndex = for_last_token(tokens, endIndex)
            end
            
            # println("\n\nstart: " * string(startIndex) * " - end: " * string(endIndex) * '\n')

            for token in tokens[startIndex:endIndex]
                temp = occursin("##", token.x) ? (temp * replace(token.x, "##"=>"")) : (temp * ' ' * token.x)
            end
            push!(chunks, temp)
            push!(lenChunks, length(tokens[startIndex:endIndex]))
            push!(sI, startIndex)
            push!(eI, endIndex)
            temp = ""

            startIndex = endIndex - 64
            endIndex = length(tokens) > (startIndex + 509) ? (startIndex + 509) : length(tokens)

            # println("\n\nstart: " * string(startIndex) * " - end: " * string(endIndex) * '\n')
        end

        # println("\n\nstart: " * string(startIndex) * " - end: " * string(endIndex) * '\n')

        for token in tokens[startIndex:endIndex]
            temp = occursin("##", token.x) ? (temp * replace(token.x, "##"=>"")) : (temp * ' ' * token.x)
        end
        push!(chunks, temp)
        push!(lenChunks, length(tokens[startIndex:endIndex]))
        push!(sI, startIndex)
        push!(eI, endIndex)

        return chunks, lenChunks, sI, eI;
    else
        return doc, length(tokens), 1, length(tokens);
    end
end
end
