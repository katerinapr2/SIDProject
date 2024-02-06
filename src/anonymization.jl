greek_tokenizer = hgf"nlpaueb/bert-base-greek-uncased-v1:tokenizer"


model_path = raw".\src\model.bson"
BSON.@load model_path cpu_model


function threshold_function(x, thres)
    if x >= thres
        x = 1
    else 
        x = 0
    end
end

    
function filter(doc)
    ner_doc = find_ner([doc], 0)

    filtered = []
    for ch in 1:length(ner_doc)
        push!(filtered, findall(x->(occursin("PERSON", x) | occursin("LOC", x) | occursin("GPE", x)), ner_doc[ch]))
    end
    return filtered
end


function predictSensitives(doc)
    doc = StringDocument(doc)
    docEncodings, tokens = preprocessDoc(doc.text, greek_tokenizer)

    # HERE: Should filter the tokens ..
    filtered = filter(doc)

    output = cpu_model(docEncodings, filtered)

    predicted = [zeros(Int8, length(tokens[:,i])) for i in 1:length(filtered)]
    st = 1
    for i in 1:length(filtered)   
        if length(filtered[i]) != 0                                                                                                              
            predicted[i][filtered[i].+1] = threshold_function.(output[st:(st+length(filtered[i])-1)], 0.5)                                                        
        end
        st = st + length(filtered[i])                                                                                                    
    end
    
    return predicted, tokens
end


function concat_arrays(pr, tok)
    allPr = vcat(pr...)
    allTok = []
    for i in 1:size(tok,2)
        append!(allTok, tok[:,i])
    end
    specialToks = findall(x->(x in ["[CLS]", "[SEP]", "[PAD]"]), allTok)
    deleteat!(allTok, specialToks)
    deleteat!(allPr, specialToks)
    return allPr, allTok
end


strip_accents_and_lowercase(x::String) = lowercase(normalize(x, stripmark=true, casefold=false))


function finalText(ps, ts, lastOfPar)
    final = ""
    partOfWord = ""
    i = 1
    punc = [".", ",", "/", ":", "!", "'"]
    while i <= length(ts)
        if ps[i] == 1 && !startswith(ts[i], "##")
            partOfWord *= ts[i]
            i += 1
            while startswith(ts[i], "##") 
                partOfWord *= replace(ts[i], "##"=>"")
                i += 1
                if i > length(ps)
                    break
                end
            end
            final = final * " " * partOfWord * " (SENSITIVE) "
            partOfWord = ""
        elseif ps[i] == 1 && startswith(ts[i], "##")
            partOfWord *= replace(ts[i], "##"=>"")
            i += 1
            while startswith(ts[i], "##")
                partOfWord *= replace(ts[i], "##"=>"")
                i += 1
                if i > length(ps)
                    break
                end
            end
            final *= partOfWord * " (SENSITIVE) "
            partOfWord = ""
        elseif ps[i] == 0
            if startswith(ts[i], "##")
                final *= replace(ts[i], "##"=>"")
            else
                if ts[i] in punc
                    final *= ts[i]
                else
                    final *= " " * ts[i]
                end
            end
            i += 1
        end

        if length(split(final)) > 3
            f = split(final)
            lastWords = f[end-3]*" "*f[end-2]*" "*f[end-1]*" "*f[end]
            
            if 1 in (lastWords .== lastOfPar)
                final *= "\r\n"
            end
        end
    end
    return final
end


function find_last_words(text::String, lastOfPar)
    par = split.(split(text, "\r\n"))

    for p in par
        if !isempty(p)
            if length(p) > 3
                push!(lastOfPar, strip_accents_and_lowercase(join(p[end-3:end], " ")))
            elseif length(p) < 4
                push!(lastOfPar, strip_accents_and_lowercase(join(p, " ")))
            end
        end
    end
    return lastOfPar
end


function anonymize(text::String)
    predicts, tokens = predictSensitives(text)

    ps, ts = concat_arrays(predicts, tokens)

    lastOfPar = []
    lastOfPar = find_last_words(text, lastOfPar)    

    strDoc = finalText(ps, ts, lastOfPar)
    open(".\\anonymized.txt", "w") do f
        write(f, strDoc)
    end
end
