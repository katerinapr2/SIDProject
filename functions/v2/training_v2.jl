using CUDA
using Flux
using Transformers
using Flux: gradient
import Flux.Optimise: update!
using Flux: DataLoader, epseltype
using Flux.Losses: _check_sizes, xlogy
using Dates: now, canonicalize
using OneHotArrays
using NeuralAttentionlib
using NeuralAttentionlib: LengthMask
using Printf
using StatsBase: mean
using EvalMetrics
using ROCCurves
using JSON
using BSON: @save
using Plots

function loss(ŷ, y, SIDLabels)    
    return Flux.binarycrossentropy(ŷ', vcat(y...))
end

function threshold_function(x, thres)
    if x >= thres
        x = 1
    else 
        x = 0
    end
end

function accuracy(x, y, SIDLabels)
    return mean(threshold_function.(x', 0.5) .== vcat(y...))
end


precision(x) = x.tp / (x.tp + x.fp)
f1_score(p, r) = (2 * p * r) / (p + r)

function concat_arrays!(outputs, y, predicts, targets)
    append!(predicts, threshold_function.(outputs', 0.5))
    append!(targets, vcat(y...))
end



function trainModel!(model, data, opt, val_data, SIDLabels, dir; n_epochs=1)
    history = Dict(
        "train_acc"=>Float64[], 
        "train_loss"=>Float64[], 
        "train_prec"=>Float64[],
        "train_rec"=>Float64[],
        "val_acc"=>Float64[], 
        "val_loss"=>Float64[], 
        "val_prec"=>Float64[],
        "val_rec"=>Float64[]
        )
    

    for epoch in 1:n_epochs
        training!(model, data, opt, SIDLabels, history)
     
        validation!(model, val_data, SIDLabels, history)

        ## Save history 
        stringdata = JSON.json(history)
        open(joinpath(dir, "history.json"), "w") do f
           write(f, stringdata)
        end

        ## Save model 
        cpu_model = model |> cpu
        output_path = joinpath(dir, "model.bson")
        @save output_path cpu_model
        println("End of epoch $epoch")
    end
    # reclaim memory
    GC.gc(true)
    println("")
    return history
end


function training!(model, data, opt, SIDLabels, history)
    epoch_train_loss = 0
    epoch_train_acc = 0
    
    totalPredicts = []
    totalTargets = []

    for (i, Xy) in enumerate(data)
        if !isempty(findall(x->!isempty(x), Xy[5]))
            input = (token=Xy[1], segment=Xy[2], attention_mask=LengthMask{1, Vector{Int32}}(Xy[3])) |> gpu
            
            l, gs = Flux.withgradient(()->loss(cpu(model(input, Xy[5] |> gpu)), Xy[4], SIDLabels), Flux.params(model.classifier))

            Flux.update!(opt, Flux.params(model.classifier), gs)
        
            outputs = model(input, Xy[5] |> gpu) |> cpu
            
            # metrics
            epoch_train_loss += l
            epoch_train_acc += accuracy(outputs, Xy[4], SIDLabels)
    
            concat_arrays!(outputs, Xy[4], totalPredicts, totalTargets)
        end

        Xy = nothing
        GC.gc(true)
    end

    cm = ConfusionMatrix(totalTargets, totalPredicts)

    push!(history["train_loss"], epoch_train_loss / length(data))
    push!(history["train_acc"], epoch_train_acc / length(data))
    push!(history["train_prec"], cm.tp / (cm.tp + cm.fp))
    push!(history["train_rec"], cm.tp / (cm.tp + cm.fn))
end


function validation!(model, data, SIDLabels, history)
    epoch_val_loss = 0
    epoch_val_acc = 0

    totalPredicts = []
    totalTargets = []

    for (i, Xy) in enumerate(data)
        if !isempty(findall(x->!isempty(x), Xy[5]))
            input = (token=Xy[1], segment=Xy[2], attention_mask=LengthMask{1, Vector{Int32}}(Xy[3])) |> gpu
            outputs = model(input, Xy[5]|> gpu) |> cpu

            epoch_val_loss += loss(outputs, Xy[4], SIDLabels)
            epoch_val_acc += accuracy(outputs, Xy[4], SIDLabels)

            concat_arrays!(outputs, Xy[4], totalPredicts, totalTargets)
        end

        Xy = nothing
        GC.gc(true)
    end

    cm = ConfusionMatrix(totalTargets, totalPredicts)

    push!(history["val_loss"], epoch_val_loss / length(data))
    push!(history["val_acc"], epoch_val_acc / length(data))
    push!(history["val_prec"], cm.tp / (cm.tp + cm.fp))
    push!(history["val_rec"], cm.tp / (cm.tp + cm.fn))
end


function testing(model, data, SIDLabels, dir)
    totalPredicts = []
    totalTargets = []
    
    for (i, Xy) in enumerate(data)
        input = (token=Xy[1], segment=Xy[2], attention_mask=LengthMask{1, Vector{Int32}}(Xy[3])) |> gpu

        outputs = model(input, Xy[5]|> gpu) |> cpu

        concat_arrays!(outputs, Xy[4], totalPredicts, totalTargets)

        Xy = nothing
        GC.gc(true)
    end

    cm = ConfusionMatrix(totalTargets, totalPredicts)
    test_acc = (cm.tp + cm.tn) / (cm.tp + cm.fp + cm.tn + cm.fn)
    test_prec = cm.tp / (cm.tp + cm.fp)
    test_rec = cm.tp / (cm.tp + cm.fn)
    test_f1 = (2 * test_prec * test_rec) / (test_prec + test_rec)

    (FPR, TPR) = roc(totalPredicts, totalTargets)
    

    println("Accuracy = $(test_acc)")
    println("Precision = $(test_prec)")
    println("Recall = $(test_rec)")
    println("F1_score = $(test_f1)")
    println("TP: $(cm.tp) - FP: $(cm.fp) - FN: $(cm.fn) - TN: $(cm.tn)")
    println("AUC = $(auc_roc(FPR, TPR))")
    
    roc_plot = plot(FPR, TPR, linetype=:steppost)
    savefig(roc_plot, joinpath(dir, "roc.png"))
end
