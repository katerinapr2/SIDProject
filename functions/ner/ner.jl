const nlp = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(nlp, pyimport("gr_nlp_toolkit"=>"Pipeline"))
end

function ner(doc, greek_tokenizer)
    labels = []
    for token in nlp("ner")(doc).tokens
        enc = Transformers.encode(greek_tokenizer, string(token.text))
        subs = Transformers.decode(greek_tokenizer, enc.token)

        if subs == 3
            push!(labels, string(token.ner))
        else
            for _ in 2:(length(subs)-1)
                push!(labels, string(token.ner))
            end
        end
    end
    return labels 
end


# find_ner() returns an array with the ner labels of the given texts. 
function find_ner(docs, overlappingChunks)
    ner_labels = Vector{Vector{String}}()
    for z in 1:length(docs)
        if overlappingChunks == 1
            chunks, _ = ToChunks.split_to_overlapping_chunks(docs[z].text, greek_tokenizer)
        else
            chunks, _ = ToChunks.split_to_chunks(docs[z].text, greek_tokenizer)
        end

        if typeof(chunks) == String 
            chunks = replace(chunks, "[UNK] "=>"")
            push!(ner_labels, ner(chunks, greek_tokenizer))  
        else
            for c in 1:length(chunks)
                chunks[c] = replace(chunks[c], "[UNK] "=>"")
                push!(ner_labels, ner(chunks[c], greek_tokenizer))  
            end
        end
    end
    return ner_labels
end